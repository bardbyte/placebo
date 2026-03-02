#!/usr/bin/env python3
"""
Conversational Analytics Agent — Tests Looker's CA API via MCP.

This agent explores what Google's Conversational Analytics can do out of the box:
- NL2SQL via Looker's semantic layer (zero custom intent classification)
- Multi-explore queries (up to 5 explores in one call)
- Schema resolution, query generation, data retrieval, NL answer synthesis

The goal is to understand CA's capabilities and limitations BEFORE we build
custom intent classification / entity resolution. If CA handles 60-80% of
queries well, our architecture changes significantly.

Usage:
    # Interactive mode
    python examples/agent_ca.py

    # Single query
    python examples/agent_ca.py "Show me total revenue by category"

    # Run evaluation suite
    python examples/agent_ca.py --eval

    # Run evaluation with JSON output
    python examples/agent_ca.py --eval --json
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lumi_llm.config import load_settings
from lumi_llm.auth import IdaaSClient
from lumi_llm.providers import GeminiProvider
from lumi_llm.mcp import MCPClient
from lumi_llm.mcp.client import MCPToolResult
from lumi_llm.agents.tool_agent import (
    create_tool_agent,
    ConsoleThinkingCallback,
    ThinkingEvent,
    ThinkingType,
    AgentState,
)


# ---------------------------------------------------------------------------
# Data models for evaluation & structured results
# ---------------------------------------------------------------------------

class QueryDifficulty(str, Enum):
    SIMPLE = "simple"           # single explore, direct field mapping
    MODERATE = "moderate"       # requires filtering, time ranges, or aggregation
    COMPLEX = "complex"         # multi-explore, ambiguous terms, joins
    ADVERSARIAL = "adversarial" # edge cases, out-of-scope, gibberish


@dataclass
class CAResponse:
    """Parsed response from the Conversational Analytics tool."""
    answer: str | None = None
    retrieval_query: dict | None = None
    data_retrieved: list[dict] | None = None
    schema_resolved: dict | None = None
    analysis: dict | None = None
    error: str | None = None
    raw: list[dict] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return self.data_retrieved is not None and len(self.data_retrieved) > 0

    @property
    def has_query(self) -> bool:
        return self.retrieval_query is not None

    @property
    def has_error(self) -> bool:
        return self.error is not None

    @property
    def fields_used(self) -> list[str]:
        """Extract fields from the generated query."""
        if not self.retrieval_query:
            return []
        return self.retrieval_query.get("fields", [])

    @property
    def filters_used(self) -> dict:
        """Extract filters from the generated query."""
        if not self.retrieval_query:
            return {}
        return self.retrieval_query.get("filters", {})

    @property
    def model_used(self) -> str | None:
        if not self.retrieval_query:
            return None
        return self.retrieval_query.get("model")

    @property
    def explore_used(self) -> str | None:
        if not self.retrieval_query:
            return None
        return self.retrieval_query.get("view")


@dataclass
class EvalResult:
    """Result of evaluating a single query against CA."""
    query: str
    difficulty: QueryDifficulty
    ca_response: CAResponse
    latency_ms: float
    # Evaluation signals
    got_answer: bool = False
    got_data: bool = False
    got_query: bool = False
    got_error: bool = False
    correct_model: bool | None = None      # None = not evaluated (no expected)
    correct_explore: bool | None = None
    correct_fields: bool | None = None
    notes: str = ""

    @property
    def passed(self) -> bool:
        """Did the CA tool produce a usable result?"""
        return self.got_answer and self.got_data and not self.got_error


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0


# ---------------------------------------------------------------------------
# CA Tool interaction layer
# ---------------------------------------------------------------------------

def parse_ca_response(raw_result: str) -> CAResponse:
    """
    Parse the raw MCP tool result from looker-conversational-analytics
    into a structured CAResponse.

    The CA tool returns an array of maps, each with a single key indicating
    the type: Answer, Retrieval Query, Data Retrieved, Schema Resolved,
    Analysis, Question, Error.
    """
    response = CAResponse()

    try:
        # The MCP result comes as a string — try to parse as JSON
        data = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except (json.JSONDecodeError, TypeError):
        # If it's not JSON, treat the whole thing as the answer text
        response.answer = str(raw_result)
        return response

    # data should be a list of dicts
    if not isinstance(data, list):
        # Could be a single dict or just text
        if isinstance(data, dict):
            data = [data]
        else:
            response.answer = str(data)
            return response

    response.raw = data

    for entry in data:
        if not isinstance(entry, dict):
            continue

        if "Answer" in entry:
            response.answer = entry["Answer"]

        elif "Retrieval Query" in entry:
            rq = entry["Retrieval Query"]
            if isinstance(rq, str):
                try:
                    rq = json.loads(rq)
                except json.JSONDecodeError:
                    pass
            response.retrieval_query = rq

        elif "Data Retrieved" in entry:
            response.data_retrieved = entry["Data Retrieved"]

        elif "Schema Resolved" in entry:
            response.schema_resolved = entry["Schema Resolved"]

        elif "Analysis" in entry:
            analysis = entry["Analysis"]
            if isinstance(analysis, str):
                try:
                    analysis = json.loads(analysis)
                except json.JSONDecodeError:
                    pass
            response.analysis = analysis

        elif "Error" in entry:
            response.error = entry["Error"]

    return response


# ---------------------------------------------------------------------------
# System prompt for the CA-powered agent
# ---------------------------------------------------------------------------

CA_SYSTEM_PROMPT = """You are Lumi, an enterprise data analyst assistant at American Express.
You help users query and understand data across the organization's data warehouse.

## Your Capabilities

You have access to two types of tools:

### Discovery Tools (for understanding what data exists)
- `get-models`: List available LookML models
- `get-explores`: List explores in a model
- `get-dimensions`: Get dimensions (attributes) in an explore
- `get-measures`: Get measures (metrics) in an explore
- `get-filters`: Get available filters
- `get-parameters`: Get parameters

### Conversational Analytics (for answering data questions)
- `ask-data-insights`: Ask a natural language question about data. This tool handles
  intent classification, entity resolution, SQL generation, and data retrieval automatically.
  You MUST provide explore_references (model + explore pairs).

### Query Tools (for precise control)
- `query`: Run a structured Looker query with specific fields and filters
- `query-sql`: Generate SQL from a Looker query

## Your Workflow

### For data questions (most common):
1. If you know which model/explore is relevant, go directly to `ask-data-insights`
2. If unsure, use `get-models` and `get-explores` first to identify the right targets
3. Pass the user's question along with explore_references to `ask-data-insights`
4. Present the results clearly with context

### For discovery questions:
- Use the discovery tools to help users understand what data is available

### For precise/complex queries:
- If `ask-data-insights` doesn't return what's needed, fall back to manual query construction
  using `get-dimensions`, `get-measures`, and `query`/`query-sql`

## Guidelines
- Always explain what you're doing and why
- When presenting data, format it as a clean table
- Show the generated query when relevant (helps users learn)
- If a query fails, explain why and suggest alternatives
- Be concise but thorough
- Proactively suggest follow-up questions the user might find useful"""


# ---------------------------------------------------------------------------
# Agent with CA-first strategy
# ---------------------------------------------------------------------------

class CAAgent:
    """
    Agent that prioritizes Conversational Analytics for NL2SQL,
    with fallback to manual query construction.

    This class wraps the standard LangGraph agent but adds:
    - CA response parsing and structured output
    - Latency tracking
    - Direct CA tool calls for evaluation (bypass LLM)
    """

    def __init__(
        self,
        llm_provider: GeminiProvider,
        mcp_client: MCPClient,
        thinking_callback: ConsoleThinkingCallback | None = None,
    ):
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client
        self.thinking_callback = thinking_callback
        self.conversation_history: list[dict] = []

        # Create the LangGraph agent for interactive mode
        self.agent = create_tool_agent(
            llm_provider=llm_provider,
            mcp_client=mcp_client,
            system_prompt=CA_SYSTEM_PROMPT,
            max_tool_calls=20,
            thinking_callback=thinking_callback,
        )

    def _find_ca_tool(self) -> str | None:
        """Find the conversational analytics tool name from MCP tools."""
        ca_names = [
            "ask-data-insights",
            "ask_data_insights",
            "conversational-analytics",
            "looker-conversational-analytics",
        ]
        for tool in self.mcp_client.tools:
            if tool.name.lower().replace("_", "-") in ca_names:
                return tool.name
        # Fallback: look for any tool with "conversational" or "insights" in name
        for tool in self.mcp_client.tools:
            name_lower = tool.name.lower()
            if "conversational" in name_lower or "insights" in name_lower:
                return tool.name
        return None

    async def discover_explores(self) -> list[dict[str, str]]:
        """
        Discover all available model/explore pairs.
        Returns list of {"model": "...", "explore": "..."} dicts.
        """
        explores = []

        # Get models
        models_result = await self.mcp_client.call_tool("get-models", {})
        if models_result.is_error:
            return explores

        try:
            models_data = json.loads(models_result.content) if isinstance(
                models_result.content, str
            ) else models_result.content
        except (json.JSONDecodeError, TypeError):
            models_data = []

        if isinstance(models_data, list):
            for model in models_data:
                model_name = model.get("name", "") if isinstance(model, dict) else str(model)
                if not model_name:
                    continue

                # Get explores for this model
                explores_result = await self.mcp_client.call_tool(
                    "get-explores", {"model": model_name}
                )
                if explores_result.is_error:
                    continue

                try:
                    explores_data = json.loads(explores_result.content) if isinstance(
                        explores_result.content, str
                    ) else explores_result.content
                except (json.JSONDecodeError, TypeError):
                    explores_data = []

                if isinstance(explores_data, list):
                    for explore in explores_data:
                        explore_name = explore.get("name", "") if isinstance(explore, dict) else str(explore)
                        if explore_name:
                            explores.append({
                                "model": model_name,
                                "explore": explore_name,
                            })

        return explores

    async def call_ca_direct(
        self,
        query: str,
        explore_references: list[dict[str, str]],
    ) -> tuple[CAResponse, float]:
        """
        Call the CA tool directly (bypassing the LLM agent loop).
        Used for evaluation and testing.

        Returns:
            Tuple of (parsed CA response, latency in ms)
        """
        ca_tool = self._find_ca_tool()
        if not ca_tool:
            return CAResponse(error="CA tool not found in MCP server"), 0.0

        start = time.perf_counter()
        result = await self.mcp_client.call_tool(ca_tool, {
            "user_query_with_context": query,
            "explore_references": explore_references,
        })
        latency_ms = (time.perf_counter() - start) * 1000

        if result.is_error:
            return CAResponse(error=str(result.content)), latency_ms

        return parse_ca_response(result.content), latency_ms

    async def chat(self, user_input: str) -> str:
        """
        Send a message through the full LangGraph agent loop.
        The agent decides which tools to use (CA or manual query).
        """
        messages = self.conversation_history + [
            {"role": "user", "content": user_input}
        ]

        initial_state: AgentState = {
            "messages": messages,
            "thinking_events": [],
            "tool_calls_made": 0,
            "final_answer": None,
        }

        print()  # Space before thinking output
        result = await self.agent.ainvoke(initial_state)
        final_answer = result.get("final_answer", "I couldn't generate a response.")

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": final_answer})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return final_answer

    def clear_history(self):
        self.conversation_history = []


# ---------------------------------------------------------------------------
# Evaluation framework
# ---------------------------------------------------------------------------

# Test queries organized by difficulty.
# These should be adapted to YOUR Looker instance's actual models/explores.
# The structure is here — replace the content with real queries once you know
# what models exist.
EVAL_QUERIES = [
    # --- SIMPLE: direct field mapping, single explore ---
    {
        "query": "What models are available?",
        "difficulty": QueryDifficulty.SIMPLE,
        "uses_ca": False,  # This is a discovery query, not CA
        "notes": "Baseline — tests model discovery, not CA",
    },
    {
        "query": "Show me total sales by product category",
        "difficulty": QueryDifficulty.SIMPLE,
        "uses_ca": True,
        "notes": "Basic aggregation with grouping",
    },
    {
        "query": "How many orders were placed last month?",
        "difficulty": QueryDifficulty.SIMPLE,
        "uses_ca": True,
        "notes": "Count with time filter — tests date understanding",
    },
    {
        "query": "What are the top 10 products by revenue?",
        "difficulty": QueryDifficulty.SIMPLE,
        "uses_ca": True,
        "notes": "Sorting + limit",
    },

    # --- MODERATE: filters, time ranges, requires field knowledge ---
    {
        "query": "Show me the monthly revenue trend for the last 6 months",
        "difficulty": QueryDifficulty.MODERATE,
        "uses_ca": True,
        "notes": "Time series — tests date dimension handling",
    },
    {
        "query": "What's the average order value by customer state?",
        "difficulty": QueryDifficulty.MODERATE,
        "uses_ca": True,
        "notes": "Computed metric + geographic dimension",
    },
    {
        "query": "Show me orders where the total is greater than 500 dollars",
        "difficulty": QueryDifficulty.MODERATE,
        "uses_ca": True,
        "notes": "Numeric filter — tests filter generation",
    },
    {
        "query": "Compare this month's sales to last month's",
        "difficulty": QueryDifficulty.MODERATE,
        "uses_ca": True,
        "notes": "Period comparison — tests date math",
    },

    # --- COMPLEX: ambiguous terms, multi-step, joins ---
    {
        "query": "Which customers haven't ordered in the last 90 days?",
        "difficulty": QueryDifficulty.COMPLEX,
        "uses_ca": True,
        "notes": "Negative filter + date math — churn analysis",
    },
    {
        "query": "What's the correlation between order frequency and average spend?",
        "difficulty": QueryDifficulty.COMPLEX,
        "uses_ca": True,
        "notes": "Computed metrics + potential code interpreter path",
    },
    {
        "query": "Show me revenue by category for the top 5 states",
        "difficulty": QueryDifficulty.COMPLEX,
        "uses_ca": True,
        "notes": "Nested aggregation — top N within grouping",
    },

    # --- ADVERSARIAL: edge cases, out-of-scope ---
    {
        "query": "What's the weather in Phoenix?",
        "difficulty": QueryDifficulty.ADVERSARIAL,
        "uses_ca": True,
        "notes": "Out-of-scope — should gracefully decline",
    },
    {
        "query": "Show me everything",
        "difficulty": QueryDifficulty.ADVERSARIAL,
        "uses_ca": True,
        "notes": "Vague query — should ask for clarification or scope down",
    },
    {
        "query": "asdfghjkl",
        "difficulty": QueryDifficulty.ADVERSARIAL,
        "uses_ca": True,
        "notes": "Gibberish — should handle gracefully",
    },
]


async def run_eval(
    agent: CAAgent,
    explore_references: list[dict[str, str]],
    output_json: bool = False,
) -> EvalSummary:
    """
    Run the evaluation suite against the CA tool.
    Calls CA directly (not through LLM agent loop) to measure
    CA's raw capability without our agent's help.
    """
    summary = EvalSummary()
    ca_queries = [q for q in EVAL_QUERIES if q.get("uses_ca", True)]
    summary.total = len(ca_queries)

    print("\n" + "=" * 70)
    print("  CONVERSATIONAL ANALYTICS EVALUATION")
    print("=" * 70)
    print(f"\n  Explores under test: {json.dumps(explore_references, indent=2)}")
    print(f"  Total queries: {summary.total}")
    print("=" * 70)

    latencies = []

    for i, test in enumerate(ca_queries, 1):
        query = test["query"]
        difficulty = test["difficulty"]
        notes = test.get("notes", "")

        print(f"\n{'─' * 70}")
        print(f"  [{i}/{summary.total}] {difficulty.value.upper()}: {query}")
        print(f"  Notes: {notes}")
        print(f"{'─' * 70}")

        ca_response, latency_ms = await agent.call_ca_direct(query, explore_references)
        latencies.append(latency_ms)

        # Build eval result
        result = EvalResult(
            query=query,
            difficulty=difficulty,
            ca_response=ca_response,
            latency_ms=latency_ms,
            got_answer=ca_response.answer is not None,
            got_data=ca_response.has_data,
            got_query=ca_response.has_query,
            got_error=ca_response.has_error,
            notes=notes,
        )

        # Print results
        status = "PASS" if result.passed else ("ERROR" if result.got_error else "FAIL")
        status_color = {"PASS": "\033[92m", "FAIL": "\033[93m", "ERROR": "\033[91m"}
        reset = "\033[0m"

        print(f"\n  Status:  {status_color.get(status, '')}{status}{reset}")
        print(f"  Latency: {latency_ms:.0f}ms")

        if ca_response.answer:
            answer_preview = ca_response.answer[:200]
            if len(ca_response.answer) > 200:
                answer_preview += "..."
            print(f"  Answer:  {answer_preview}")

        if ca_response.has_query:
            print(f"  Model:   {ca_response.model_used}")
            print(f"  Explore: {ca_response.explore_used}")
            print(f"  Fields:  {ca_response.fields_used[:5]}{'...' if len(ca_response.fields_used) > 5 else ''}")
            if ca_response.filters_used:
                print(f"  Filters: {json.dumps(ca_response.filters_used)}")

        if ca_response.has_data:
            print(f"  Rows:    {len(ca_response.data_retrieved)}")

        if ca_response.has_error:
            print(f"  Error:   {ca_response.error[:200]}")

        # Track results
        if result.passed:
            summary.passed += 1
        elif result.got_error:
            summary.errors += 1
        else:
            summary.failed += 1

        summary.results.append(result)

        # Track by difficulty
        diff_key = difficulty.value
        if diff_key not in summary.by_difficulty:
            summary.by_difficulty[diff_key] = {"total": 0, "passed": 0, "failed": 0, "errors": 0}
        summary.by_difficulty[diff_key]["total"] += 1
        if result.passed:
            summary.by_difficulty[diff_key]["passed"] += 1
        elif result.got_error:
            summary.by_difficulty[diff_key]["errors"] += 1
        else:
            summary.by_difficulty[diff_key]["failed"] += 1

    # Summary
    summary.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n  Total:     {summary.total}")
    print(f"  Passed:    \033[92m{summary.passed}\033[0m ({summary.pass_rate:.0f}%)")
    print(f"  Failed:    \033[93m{summary.failed}\033[0m")
    print(f"  Errors:    \033[91m{summary.errors}\033[0m")
    print(f"  Avg Latency: {summary.avg_latency_ms:.0f}ms")

    print(f"\n  By Difficulty:")
    for diff, stats in summary.by_difficulty.items():
        rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        print(f"    {diff:14s}  {stats['passed']}/{stats['total']} passed ({rate:.0f}%)")

    print("=" * 70)

    # JSON output for CI/pipeline integration
    if output_json:
        json_path = Path(__file__).parent.parent / "eval_results_ca.json"
        json_output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "errors": summary.errors,
            "pass_rate": summary.pass_rate,
            "avg_latency_ms": summary.avg_latency_ms,
            "by_difficulty": summary.by_difficulty,
            "results": [
                {
                    "query": r.query,
                    "difficulty": r.difficulty.value,
                    "passed": r.passed,
                    "latency_ms": r.latency_ms,
                    "got_answer": r.got_answer,
                    "got_data": r.got_data,
                    "got_query": r.got_query,
                    "got_error": r.got_error,
                    "answer_preview": (r.ca_response.answer or "")[:200],
                    "model_used": r.ca_response.model_used,
                    "explore_used": r.ca_response.explore_used,
                    "fields_used": r.ca_response.fields_used,
                    "filters_used": r.ca_response.filters_used,
                    "error": r.ca_response.error,
                    "notes": r.notes,
                }
                for r in summary.results
            ],
        }
        with open(json_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\n  Results written to: {json_path}")

    return summary


# ---------------------------------------------------------------------------
# Interactive chat
# ---------------------------------------------------------------------------

async def interactive_chat(agent: CAAgent):
    """Run interactive chat session."""
    print("""
╭─────────────────────────────────────────────────────────────╮
│                                                             │
│          LUMI — Conversational Analytics Agent              │
│                                                             │
│    Powered by Looker CA API + Gemini via MCP                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  /tools    - List available MCP tools                       │
│  /explores - Show discovered model/explore pairs            │
│  /ca <q>   - Call CA directly (bypass agent, show raw)      │
│  /clear    - Clear conversation history                     │
│  /help     - Show this help                                 │
│  /quit     - Exit                                           │
╰─────────────────────────────────────────────────────────────╯
""")

    # Discover explores on startup
    print("Discovering available explores...")
    explores = await agent.discover_explores()
    if explores:
        print(f"Found {len(explores)} explores:")
        for exp in explores[:10]:
            print(f"  - {exp['model']}.{exp['explore']}")
        if len(explores) > 10:
            print(f"  ... and {len(explores) - 10} more")
    else:
        print("  No explores found. Check MCP server connection.")

    print("\nType your question or command.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            # Commands
            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break

            elif user_input.lower() == "/tools":
                print(f"\nAvailable tools ({len(agent.mcp_client.tools)}):")
                for tool in agent.mcp_client.tools:
                    desc = tool.description.strip().split("\n")[0][:60]
                    print(f"  - {tool.name}: {desc}")
                print()
                continue

            elif user_input.lower() == "/explores":
                explores = await agent.discover_explores()
                print(f"\nExplores ({len(explores)}):")
                for exp in explores:
                    print(f"  - {exp['model']}.{exp['explore']}")
                print()
                continue

            elif user_input.lower().startswith("/ca "):
                # Direct CA call — bypass agent, show raw response
                query = user_input[4:].strip()
                if not explores:
                    print("No explores discovered. Run /explores first.")
                    continue
                print(f"\nCalling CA directly with {len(explores)} explores...")
                ca_response, latency = await agent.call_ca_direct(query, explores[:5])
                print(f"\n{'─' * 60}")
                print(f"Latency: {latency:.0f}ms")
                if ca_response.answer:
                    print(f"Answer: {ca_response.answer}")
                if ca_response.has_query:
                    print(f"Query: {json.dumps(ca_response.retrieval_query, indent=2)}")
                if ca_response.has_data:
                    print(f"Data ({len(ca_response.data_retrieved)} rows):")
                    for row in ca_response.data_retrieved[:5]:
                        print(f"  {row}")
                    if len(ca_response.data_retrieved) > 5:
                        print(f"  ... {len(ca_response.data_retrieved) - 5} more rows")
                if ca_response.has_error:
                    print(f"Error: {ca_response.error}")
                if not any([ca_response.answer, ca_response.has_query, ca_response.has_data, ca_response.has_error]):
                    print(f"Raw response: {json.dumps(ca_response.raw, indent=2)}")
                print(f"{'─' * 60}\n")
                continue

            elif user_input.lower() == "/clear":
                agent.clear_history()
                print("\n[Conversation history cleared]\n")
                continue

            elif user_input.lower() == "/help":
                print("""
Commands:
  /tools    - List available MCP tools
  /explores - Show discovered model/explore pairs
  /ca <q>   - Call CA directly (bypass agent, show raw response)
  /clear    - Clear conversation history
  /quit     - Exit
""")
                continue

            # Regular chat through agent
            try:
                response = await agent.chat(user_input)
                print(f"\n{'─' * 60}")
                print(f"Lumi: {response}")
                print(f"{'─' * 60}\n")
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                print()

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    """Main entry point."""
    args = sys.argv[1:]
    run_evaluation = "--eval" in args
    output_json = "--json" in args
    single_query = None

    # Extract non-flag arguments as query
    query_args = [a for a in args if not a.startswith("--")]
    if query_args:
        single_query = " ".join(query_args)

    # --- Setup ---
    print("=" * 60)
    print("  Lumi — Conversational Analytics Agent")
    print("=" * 60)

    print("\n[1/4] Loading configuration...")
    try:
        config_path = Path(__file__).parent.parent / "config.yaml"
        env_path = Path(__file__).parent.parent / ".env"
        settings = load_settings(config_path=config_path, env_path=env_path)
        print("      OK")
    except FileNotFoundError as e:
        print(f"      Error: {e}")
        return

    print("[2/4] Initializing authentication...")
    auth_client = IdaaSClient(settings.idaas)
    llm_provider = GeminiProvider(settings.llm, auth_client)
    print("      OK")

    print("[3/4] Connecting to MCP server...")
    looker_config = settings.mcp.servers.get("looker")
    if not looker_config:
        print("      Error: 'looker' MCP server not configured in config.yaml")
        return

    mcp_client = MCPClient(looker_config)
    try:
        await mcp_client.connect()
        tool_names = [t.name for t in mcp_client.tools]
        print(f"      Connected — {len(mcp_client.tools)} tools: {tool_names}")
    except Exception as e:
        print(f"      Error: {e}")
        print(f"      Ensure MCP Toolbox is running at {looker_config.url}")
        print("      Start with: ./toolbox --tools-file tools.yaml")
        return

    # Check for CA tool
    print("[4/4] Checking for Conversational Analytics tool...")
    thinking_callback = ConsoleThinkingCallback(use_rich=True)
    agent = CAAgent(llm_provider, mcp_client, thinking_callback)
    ca_tool = agent._find_ca_tool()

    if ca_tool:
        print(f"      Found CA tool: {ca_tool}")
    else:
        print("      CA tool NOT found. Available tools:")
        for t in mcp_client.tools:
            print(f"        - {t.name}")
        print("\n      To enable CA, add to tools.yaml:")
        print("        ask-data-insights:")
        print("          kind: looker-conversational-analytics")
        print("          source: my-looker")
        print("      And add project/location to the looker source.")
        if not run_evaluation:
            print("\n      Continuing without CA (discovery + manual query tools still work)...")

    # --- Run mode ---
    if run_evaluation:
        if not ca_tool:
            print("\n      Cannot run evaluation without CA tool. Exiting.")
            await mcp_client.disconnect()
            return

        # Discover explores for eval
        print("\nDiscovering explores for evaluation...")
        explores = await agent.discover_explores()
        if not explores:
            print("No explores found. Cannot run evaluation.")
            await mcp_client.disconnect()
            return

        print(f"Found {len(explores)} explores. Using first 5 for eval.")
        await run_eval(agent, explores[:5], output_json=output_json)

    elif single_query:
        # Single query mode
        if ca_tool:
            explores = await agent.discover_explores()
            if explores:
                print(f"\nDirect CA call: \"{single_query}\"")
                ca_response, latency = await agent.call_ca_direct(
                    single_query, explores[:5]
                )
                print(f"\nLatency: {latency:.0f}ms")
                if ca_response.answer:
                    print(f"Answer: {ca_response.answer}")
                if ca_response.has_query:
                    print(f"Query: {json.dumps(ca_response.retrieval_query, indent=2)}")
                if ca_response.has_data:
                    print(f"Data: {len(ca_response.data_retrieved)} rows")
                    for row in ca_response.data_retrieved[:10]:
                        print(f"  {row}")
                if ca_response.has_error:
                    print(f"Error: {ca_response.error}")
            else:
                # Fall back to agent
                response = await agent.chat(single_query)
                print(f"\n{response}")
        else:
            response = await agent.chat(single_query)
            print(f"\n{response}")

    else:
        # Interactive mode
        await interactive_chat(agent)

    await mcp_client.disconnect()
    print("\n[Disconnected]")


if __name__ == "__main__":
    asyncio.run(main())
