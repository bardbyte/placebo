#!/usr/bin/env python3
"""
Lumi Evaluation Pipeline — Schema-Aware, Auto-Generated, Reusable.

This pipeline:
1. Crawls the Looker instance to discover all models, explores, dimensions, measures
2. Uses the LLM to generate test questions grounded in the ACTUAL schema
3. Runs those questions against the system under test (CA API or full agent)
4. Scores results with production-grade metrics
5. Outputs structured JSON for CI/CD integration and trend tracking

Designed to be reusable from PoC (March 2026) through production (June 2026+).
The system-under-test is pluggable — today it's CA, tomorrow it's the full Lumi agent.

Usage:
    # Full pipeline: discover → generate → evaluate → report
    python examples/eval_pipeline.py

    # Skip question generation, reuse last generated set
    python examples/eval_pipeline.py --reuse

    # Generate questions only (no eval)
    python examples/eval_pipeline.py --generate-only

    # Run eval on a saved question set
    python examples/eval_pipeline.py --questions eval_questions.json

    # Control question count per difficulty
    python examples/eval_pipeline.py --count 10

    # Verbose output
    python examples/eval_pipeline.py -v
"""

import asyncio
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).parent.parent))

from lumi_llm.config import load_settings
from lumi_llm.auth import IdaaSClient
from lumi_llm.providers import GeminiProvider
from lumi_llm.mcp import MCPClient
from examples.agent_ca import CAAgent, CAResponse, parse_ca_response


# ===========================================================================
# Schema Discovery
# ===========================================================================

@dataclass
class FieldInfo:
    """A dimension or measure in an explore."""
    name: str
    label: str
    field_type: str  # "dimension" or "measure"
    data_type: str   # "string", "number", "date", "yesno", etc.
    description: str
    tags: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ExploreInfo:
    """An explore with its fields."""
    model: str
    name: str
    label: str
    description: str
    dimensions: list[FieldInfo] = field(default_factory=list)
    measures: list[FieldInfo] = field(default_factory=list)

    @property
    def all_fields(self) -> list[FieldInfo]:
        return self.dimensions + self.measures

    @property
    def date_dimensions(self) -> list[FieldInfo]:
        return [d for d in self.dimensions if d.data_type in ("date", "date_time", "date_date", "date_week", "date_month", "date_quarter", "date_year")]

    @property
    def numeric_measures(self) -> list[FieldInfo]:
        return [m for m in self.measures if m.data_type in ("number", "count", "sum", "average", "int")]

    @property
    def string_dimensions(self) -> list[FieldInfo]:
        return [d for d in self.dimensions if d.data_type in ("string", "zipcode")]

    def schema_summary(self, max_fields: int = 30) -> str:
        """Generate a concise schema summary for LLM prompting."""
        lines = [f"Model: {self.model}, Explore: {self.name}"]
        if self.description:
            lines.append(f"Description: {self.description}")

        lines.append(f"\nDimensions ({len(self.dimensions)}):")
        for d in self.dimensions[:max_fields]:
            desc = f" — {d.description}" if d.description else ""
            lines.append(f"  - {d.name} ({d.data_type}){desc}")
        if len(self.dimensions) > max_fields:
            lines.append(f"  ... and {len(self.dimensions) - max_fields} more")

        lines.append(f"\nMeasures ({len(self.measures)}):")
        for m in self.measures[:max_fields]:
            desc = f" — {m.description}" if m.description else ""
            lines.append(f"  - {m.name} ({m.data_type}){desc}")
        if len(self.measures) > max_fields:
            lines.append(f"  ... and {len(self.measures) - max_fields} more")

        return "\n".join(lines)


@dataclass
class LookerSchema:
    """Complete discovered schema from a Looker instance."""
    explores: list[ExploreInfo] = field(default_factory=list)
    discovered_at: str = ""
    instance_fingerprint: str = ""

    @property
    def total_dimensions(self) -> int:
        return sum(len(e.dimensions) for e in self.explores)

    @property
    def total_measures(self) -> int:
        return sum(len(e.measures) for e in self.explores)

    @property
    def models(self) -> list[str]:
        return list(set(e.model for e in self.explores))

    def fingerprint(self) -> str:
        """Generate a fingerprint of the schema for cache invalidation."""
        content = json.dumps(
            [{"model": e.model, "explore": e.name,
              "dims": len(e.dimensions), "measures": len(e.measures)}
             for e in self.explores],
            sort_keys=True,
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "discovered_at": self.discovered_at,
            "instance_fingerprint": self.instance_fingerprint,
            "models": self.models,
            "explores": [
                {
                    "model": e.model,
                    "name": e.name,
                    "label": e.label,
                    "description": e.description,
                    "dimension_count": len(e.dimensions),
                    "measure_count": len(e.measures),
                    "dimensions": [
                        {"name": d.name, "label": d.label, "type": d.data_type,
                         "description": d.description, "tags": d.tags}
                        for d in e.dimensions
                    ],
                    "measures": [
                        {"name": m.name, "label": m.label, "type": m.data_type,
                         "description": m.description, "tags": m.tags}
                        for m in e.measures
                    ],
                }
                for e in self.explores
            ],
        }


async def discover_schema(mcp_client: MCPClient, verbose: bool = False) -> LookerSchema:
    """
    Crawl the Looker instance via MCP to discover the full schema.
    Returns structured LookerSchema with all models, explores, dimensions, measures.
    """
    schema = LookerSchema(discovered_at=datetime.now(timezone.utc).isoformat())

    # 1. Get models
    if verbose:
        print("  Discovering models...")
    models_result = await mcp_client.call_tool("get-models", {})
    if models_result.is_error:
        print(f"  Error getting models: {models_result.content}")
        return schema

    try:
        models_data = json.loads(models_result.content) if isinstance(
            models_result.content, str) else models_result.content
    except (json.JSONDecodeError, TypeError):
        models_data = []

    if not isinstance(models_data, list):
        models_data = []

    model_names = []
    for m in models_data:
        name = m.get("name", "") if isinstance(m, dict) else str(m)
        if name:
            model_names.append(name)

    if verbose:
        print(f"  Found {len(model_names)} models: {model_names}")

    # 2. For each model, get explores
    for model_name in model_names:
        if verbose:
            print(f"  Discovering explores for model '{model_name}'...")

        explores_result = await mcp_client.call_tool("get-explores", {"model": model_name})
        if explores_result.is_error:
            continue

        try:
            explores_data = json.loads(explores_result.content) if isinstance(
                explores_result.content, str) else explores_result.content
        except (json.JSONDecodeError, TypeError):
            explores_data = []

        if not isinstance(explores_data, list):
            continues = []

        for exp in explores_data:
            if not isinstance(exp, dict):
                continue
            explore_name = exp.get("name", "")
            if not explore_name:
                continue

            explore_info = ExploreInfo(
                model=model_name,
                name=explore_name,
                label=exp.get("label", explore_name),
                description=exp.get("description", ""),
            )

            # 3. Get dimensions
            if verbose:
                print(f"    Getting fields for {model_name}.{explore_name}...")

            dims_result = await mcp_client.call_tool("get-dimensions", {
                "model": model_name, "explore": explore_name,
            })
            if not dims_result.is_error:
                try:
                    dims_data = json.loads(dims_result.content) if isinstance(
                        dims_result.content, str) else dims_result.content
                except (json.JSONDecodeError, TypeError):
                    dims_data = []

                if isinstance(dims_data, list):
                    for d in dims_data:
                        if not isinstance(d, dict):
                            continue
                        explore_info.dimensions.append(FieldInfo(
                            name=d.get("name", ""),
                            label=d.get("label", ""),
                            field_type="dimension",
                            data_type=d.get("type", "string"),
                            description=d.get("description", ""),
                            tags=d.get("tags", []),
                            suggestions=d.get("suggestions", []),
                        ))

            # 4. Get measures
            measures_result = await mcp_client.call_tool("get-measures", {
                "model": model_name, "explore": explore_name,
            })
            if not measures_result.is_error:
                try:
                    measures_data = json.loads(measures_result.content) if isinstance(
                        measures_result.content, str) else measures_result.content
                except (json.JSONDecodeError, TypeError):
                    measures_data = []

                if isinstance(measures_data, list):
                    for m in measures_data:
                        if not isinstance(m, dict):
                            continue
                        explore_info.measures.append(FieldInfo(
                            name=m.get("name", ""),
                            label=m.get("label", ""),
                            field_type="measure",
                            data_type=m.get("type", "number"),
                            description=m.get("description", ""),
                            tags=m.get("tags", []),
                        ))

            schema.explores.append(explore_info)
            if verbose:
                print(f"      {len(explore_info.dimensions)} dims, {len(explore_info.measures)} measures")

    schema.instance_fingerprint = schema.fingerprint()

    if verbose:
        print(f"\n  Schema discovery complete:")
        print(f"    Models: {len(schema.models)}")
        print(f"    Explores: {len(schema.explores)}")
        print(f"    Dimensions: {schema.total_dimensions}")
        print(f"    Measures: {schema.total_measures}")
        print(f"    Fingerprint: {schema.instance_fingerprint}")

    return schema


# ===========================================================================
# Question Generation
# ===========================================================================

class QueryDifficulty(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVERSARIAL = "adversarial"


@dataclass
class GeneratedQuestion:
    """A test question generated from the schema."""
    id: str
    query: str
    difficulty: QueryDifficulty
    category: str                          # What aspect it tests
    target_model: str
    target_explore: str
    expected_dimensions: list[str] = field(default_factory=list)
    expected_measures: list[str] = field(default_factory=list)
    expected_filters: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "difficulty": self.difficulty.value,
            "category": self.category,
            "target_model": self.target_model,
            "target_explore": self.target_explore,
            "expected_dimensions": self.expected_dimensions,
            "expected_measures": self.expected_measures,
            "expected_filters": self.expected_filters,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GeneratedQuestion":
        return cls(
            id=data["id"],
            query=data["query"],
            difficulty=QueryDifficulty(data["difficulty"]),
            category=data.get("category", ""),
            target_model=data.get("target_model", ""),
            target_explore=data.get("target_explore", ""),
            expected_dimensions=data.get("expected_dimensions", []),
            expected_measures=data.get("expected_measures", []),
            expected_filters=data.get("expected_filters", {}),
            notes=data.get("notes", ""),
        )


QUESTION_GEN_PROMPT = """You are generating test questions for evaluating an NL2SQL system.
The system translates natural language questions into SQL queries over a Looker semantic layer.

Below is the ACTUAL schema from a Looker instance. Generate test questions that a real
business user would ask, grounded in the REAL fields available.

{schema}

---

Generate exactly {count} questions for difficulty level: **{difficulty}**

Difficulty definitions:
- **simple**: Direct field lookup or single aggregation. Uses 1-2 fields. No filters or simple equality filter.
  Example patterns: "What is the total X?", "Show me Y by Z", "List all X"
- **moderate**: Requires filtering, date ranges, sorting, or 2-3 field combinations.
  Example patterns: "Show me X for last month", "Top 10 Y by Z", "Average X by Y where Z > N"
- **complex**: Requires multiple fields, complex filters, computed comparisons, or reasoning about relationships.
  Example patterns: "Compare X this month vs last month", "Which Y has the highest growth in X?",
  "Show me X by Y for the top 5 Z"
- **adversarial**: Edge cases that test graceful failure. Include 2-3 of each:
  - Out-of-scope questions (weather, sports, unrelated topics)
  - Vague/ambiguous questions ("show me data", "tell me about things")
  - Questions using wrong field names or nonexistent fields
  - Questions that mix fields from incompatible explores

For each question, specify:
1. The natural language query a user would type
2. Which model and explore should be used
3. Which dimensions and measures should appear in the result
4. Any filters that should be applied
5. A category tag describing what aspect of the system it tests
6. Brief notes on what makes this question interesting for evaluation

Respond in this exact JSON format (array of objects):
```json
[
  {{
    "query": "What is the total revenue by product category?",
    "target_model": "ecommerce",
    "target_explore": "order_items",
    "expected_dimensions": ["products.category"],
    "expected_measures": ["order_items.total_revenue"],
    "expected_filters": {{}},
    "category": "basic_aggregation",
    "notes": "Simple group-by with one measure"
  }}
]
```

IMPORTANT:
- Use ACTUAL field names from the schema above (e.g., "explore_name.field_name")
- For adversarial questions, expected_dimensions and expected_measures can be empty
- Questions must be phrased as a business user would ask (no SQL, no field names)
- Vary the question phrasing — don't start every question with "Show me"
- For date filters, use relative terms like "last month", "last 7 days", "this quarter"
"""


async def generate_questions(
    llm_provider: GeminiProvider,
    schema: LookerSchema,
    count_per_difficulty: int = 5,
    verbose: bool = False,
) -> list[GeneratedQuestion]:
    """
    Use the LLM to generate test questions based on the discovered schema.
    """
    questions = []
    question_id = 0

    for difficulty in QueryDifficulty:
        if verbose:
            print(f"  Generating {count_per_difficulty} {difficulty.value} questions...")

        # Build schema context — include all explores but cap field listings
        schema_parts = []
        for explore in schema.explores:
            schema_parts.append(explore.schema_summary(max_fields=20))
        schema_text = "\n\n---\n\n".join(schema_parts)

        prompt = QUESTION_GEN_PROMPT.format(
            schema=schema_text,
            count=count_per_difficulty,
            difficulty=difficulty.value,
        )

        response = await llm_provider.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096,
        )

        if not response.content:
            print(f"  Warning: empty response for {difficulty.value}")
            continue

        # Parse the JSON from the response
        raw = response.content.strip()
        # Extract JSON array from markdown code block if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  Warning: failed to parse LLM response for {difficulty.value}: {e}")
            if verbose:
                print(f"  Raw response: {raw[:500]}")
            continue

        if not isinstance(parsed, list):
            parsed = [parsed]

        for item in parsed:
            question_id += 1
            q = GeneratedQuestion(
                id=f"{difficulty.value}_{question_id:03d}",
                query=item.get("query", ""),
                difficulty=difficulty,
                category=item.get("category", "unknown"),
                target_model=item.get("target_model", ""),
                target_explore=item.get("target_explore", ""),
                expected_dimensions=item.get("expected_dimensions", []),
                expected_measures=item.get("expected_measures", []),
                expected_filters=item.get("expected_filters", {}),
                notes=item.get("notes", ""),
            )
            if q.query:
                questions.append(q)

        if verbose:
            print(f"    Generated {len([q for q in questions if q.difficulty == difficulty])} questions")

    return questions


# ===========================================================================
# Evaluation Metrics — Production Grade
# ===========================================================================

@dataclass
class EvalMetrics:
    """
    Comprehensive metrics for evaluating an NL2SQL system.
    Designed to be reusable from PoC through production.
    """
    # --- Functional metrics ---
    total_queries: int = 0
    successful_queries: int = 0          # Got answer + data + no error
    failed_queries: int = 0              # Got answer but no data
    error_queries: int = 0               # System error
    empty_result_queries: int = 0        # Query ran but returned 0 rows

    # --- Accuracy metrics ---
    correct_model_count: int = 0         # Picked the right model
    correct_explore_count: int = 0       # Picked the right explore
    correct_fields_count: int = 0        # Used expected dimensions/measures
    partial_fields_count: int = 0        # Used some expected fields
    has_expected_count: int = 0          # Queries where we have expected values

    # --- Quality metrics ---
    answer_provided_count: int = 0       # Returned an NL answer
    query_generated_count: int = 0       # Generated a Looker query
    filter_applied_count: int = 0        # Applied at least one filter
    correct_filter_count: int = 0        # Applied the expected filters

    # --- Latency metrics ---
    latencies_ms: list[float] = field(default_factory=list)

    # --- Per-difficulty breakdown ---
    by_difficulty: dict[str, dict] = field(default_factory=dict)

    # --- Per-category breakdown ---
    by_category: dict[str, dict] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0

    @property
    def model_accuracy(self) -> float:
        return (self.correct_model_count / self.has_expected_count * 100) if self.has_expected_count > 0 else 0

    @property
    def explore_accuracy(self) -> float:
        return (self.correct_explore_count / self.has_expected_count * 100) if self.has_expected_count > 0 else 0

    @property
    def field_accuracy(self) -> float:
        return (self.correct_fields_count / self.has_expected_count * 100) if self.has_expected_count > 0 else 0

    @property
    def answer_rate(self) -> float:
        return (self.answer_provided_count / self.total_queries * 100) if self.total_queries > 0 else 0

    @property
    def latency_p50(self) -> float:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        return s[len(s) // 2]

    @property
    def latency_p90(self) -> float:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.9)]

    @property
    def latency_p99(self) -> float:
        if not self.latencies_ms:
            return 0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.99)]

    @property
    def latency_avg(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0

    def _ensure_bucket(self, bucket_dict: dict, key: str):
        if key not in bucket_dict:
            bucket_dict[key] = {
                "total": 0, "success": 0, "failed": 0, "error": 0,
                "correct_model": 0, "correct_explore": 0, "correct_fields": 0,
                "has_expected": 0, "latencies": [],
            }

    def record(
        self,
        question: GeneratedQuestion,
        ca_response: CAResponse,
        latency_ms: float,
    ):
        """Record a single evaluation result."""
        self.total_queries += 1
        self.latencies_ms.append(latency_ms)

        got_answer = ca_response.answer is not None
        got_data = ca_response.has_data
        got_error = ca_response.has_error
        got_query = ca_response.has_query

        if got_answer:
            self.answer_provided_count += 1
        if got_query:
            self.query_generated_count += 1
        if ca_response.filters_used:
            self.filter_applied_count += 1

        # Success/failure classification
        if got_error:
            self.error_queries += 1
            status = "error"
        elif got_answer and got_data:
            self.successful_queries += 1
            status = "success"
        elif got_answer and got_query and not got_data:
            # Query ran but empty results — might be correct behavior
            self.empty_result_queries += 1
            status = "success"  # Count as success — the system worked, data was just empty
            self.successful_queries += 1
        else:
            self.failed_queries += 1
            status = "failed"

        # Accuracy checks (only for non-adversarial with expected values)
        has_expected = (
            question.difficulty != QueryDifficulty.ADVERSARIAL
            and (question.expected_dimensions or question.expected_measures)
        )
        if has_expected:
            self.has_expected_count += 1

            # Model accuracy
            if ca_response.model_used and ca_response.model_used == question.target_model:
                self.correct_model_count += 1
                model_correct = True
            else:
                model_correct = False

            # Explore accuracy
            if ca_response.explore_used and ca_response.explore_used == question.target_explore:
                self.correct_explore_count += 1
                explore_correct = True
            else:
                explore_correct = False

            # Field accuracy — check if expected fields are subset of used fields
            used_fields = set(ca_response.fields_used)
            expected_fields = set(question.expected_dimensions + question.expected_measures)
            if expected_fields and expected_fields.issubset(used_fields):
                self.correct_fields_count += 1
                fields_correct = True
            elif expected_fields and expected_fields.intersection(used_fields):
                self.partial_fields_count += 1
                fields_correct = False
            else:
                fields_correct = False

            # Filter accuracy
            if question.expected_filters and ca_response.filters_used:
                expected_filter_keys = set(question.expected_filters.keys())
                actual_filter_keys = set(ca_response.filters_used.keys())
                if expected_filter_keys.issubset(actual_filter_keys):
                    self.correct_filter_count += 1
        else:
            model_correct = None
            explore_correct = None
            fields_correct = None

        # Per-difficulty tracking
        diff_key = question.difficulty.value
        self._ensure_bucket(self.by_difficulty, diff_key)
        b = self.by_difficulty[diff_key]
        b["total"] += 1
        b["latencies"].append(latency_ms)
        if status == "success":
            b["success"] += 1
        elif status == "error":
            b["error"] += 1
        else:
            b["failed"] += 1
        if has_expected:
            b["has_expected"] += 1
            if model_correct:
                b["correct_model"] += 1
            if explore_correct:
                b["correct_explore"] += 1
            if fields_correct:
                b["correct_fields"] += 1

        # Per-category tracking
        cat_key = question.category or "uncategorized"
        self._ensure_bucket(self.by_category, cat_key)
        c = self.by_category[cat_key]
        c["total"] += 1
        c["latencies"].append(latency_ms)
        if status == "success":
            c["success"] += 1
        elif status == "error":
            c["error"] += 1
        else:
            c["failed"] += 1

    def to_dict(self) -> dict:
        """Export metrics as a dict for JSON serialization."""
        by_diff_clean = {}
        for k, v in self.by_difficulty.items():
            total = v["total"]
            by_diff_clean[k] = {
                "total": total,
                "success": v["success"],
                "failed": v["failed"],
                "error": v["error"],
                "success_rate": (v["success"] / total * 100) if total > 0 else 0,
                "model_accuracy": (v["correct_model"] / v["has_expected"] * 100) if v["has_expected"] > 0 else None,
                "explore_accuracy": (v["correct_explore"] / v["has_expected"] * 100) if v["has_expected"] > 0 else None,
                "field_accuracy": (v["correct_fields"] / v["has_expected"] * 100) if v["has_expected"] > 0 else None,
                "latency_avg_ms": sum(v["latencies"]) / len(v["latencies"]) if v["latencies"] else 0,
            }

        by_cat_clean = {}
        for k, v in self.by_category.items():
            total = v["total"]
            by_cat_clean[k] = {
                "total": total,
                "success": v["success"],
                "failed": v["failed"],
                "success_rate": (v["success"] / total * 100) if total > 0 else 0,
            }

        return {
            "summary": {
                "total_queries": self.total_queries,
                "successful": self.successful_queries,
                "failed": self.failed_queries,
                "errors": self.error_queries,
                "empty_results": self.empty_result_queries,
                "success_rate": round(self.success_rate, 1),
                "answer_rate": round(self.answer_rate, 1),
            },
            "accuracy": {
                "model_accuracy": round(self.model_accuracy, 1),
                "explore_accuracy": round(self.explore_accuracy, 1),
                "field_accuracy": round(self.field_accuracy, 1),
                "queries_with_expectations": self.has_expected_count,
                "queries_with_filters_applied": self.filter_applied_count,
                "correct_filters": self.correct_filter_count,
            },
            "latency": {
                "avg_ms": round(self.latency_avg, 0),
                "p50_ms": round(self.latency_p50, 0),
                "p90_ms": round(self.latency_p90, 0),
                "p99_ms": round(self.latency_p99, 0),
            },
            "by_difficulty": by_diff_clean,
            "by_category": by_cat_clean,
        }

    def print_report(self):
        """Print a human-readable report."""
        d = self.to_dict()
        s = d["summary"]
        a = d["accuracy"]
        l = d["latency"]

        print("\n" + "=" * 70)
        print("  EVALUATION REPORT")
        print("=" * 70)

        print(f"\n  Functional Metrics")
        print(f"  {'─' * 40}")
        print(f"  Total queries:    {s['total_queries']}")
        print(f"  Successful:       \033[92m{s['successful']}\033[0m ({s['success_rate']}%)")
        print(f"  Failed:           \033[93m{s['failed']}\033[0m")
        print(f"  Errors:           \033[91m{s['errors']}\033[0m")
        print(f"  Empty results:    {s['empty_results']}")
        print(f"  Answer rate:      {s['answer_rate']}%")

        print(f"\n  Accuracy Metrics (on {a['queries_with_expectations']} queries with expectations)")
        print(f"  {'─' * 40}")
        print(f"  Model accuracy:   {a['model_accuracy']}%")
        print(f"  Explore accuracy: {a['explore_accuracy']}%")
        print(f"  Field accuracy:   {a['field_accuracy']}%")
        print(f"  Filter accuracy:  {a['correct_filters']}/{a['queries_with_filters_applied']} correct")

        print(f"\n  Latency")
        print(f"  {'─' * 40}")
        print(f"  Avg:  {l['avg_ms']:.0f}ms")
        print(f"  P50:  {l['p50_ms']:.0f}ms")
        print(f"  P90:  {l['p90_ms']:.0f}ms")
        print(f"  P99:  {l['p99_ms']:.0f}ms")

        print(f"\n  By Difficulty")
        print(f"  {'─' * 40}")
        for diff, stats in d["by_difficulty"].items():
            sr = stats["success_rate"]
            ma = f", model={stats['model_accuracy']:.0f}%" if stats["model_accuracy"] is not None else ""
            ea = f", explore={stats['explore_accuracy']:.0f}%" if stats["explore_accuracy"] is not None else ""
            print(f"  {diff:14s}  {stats['success']}/{stats['total']} ({sr:.0f}%){ma}{ea}")

        if d["by_category"]:
            print(f"\n  By Category")
            print(f"  {'─' * 40}")
            for cat, stats in sorted(d["by_category"].items()):
                sr = stats["success_rate"]
                print(f"  {cat:25s}  {stats['success']}/{stats['total']} ({sr:.0f}%)")

        print("\n" + "=" * 70)


# ===========================================================================
# Evaluation Runner
# ===========================================================================

@dataclass
class DetailedResult:
    """Full result for a single evaluated question."""
    question: GeneratedQuestion
    ca_response: CAResponse
    latency_ms: float
    status: str  # "success", "failed", "error"

    def to_dict(self) -> dict:
        return {
            "question": self.question.to_dict(),
            "status": self.status,
            "latency_ms": round(self.latency_ms, 1),
            "answer": self.ca_response.answer,
            "model_used": self.ca_response.model_used,
            "explore_used": self.ca_response.explore_used,
            "fields_used": self.ca_response.fields_used,
            "filters_used": self.ca_response.filters_used,
            "data_row_count": len(self.ca_response.data_retrieved) if self.ca_response.data_retrieved else 0,
            "error": self.ca_response.error,
        }


async def run_evaluation(
    agent: CAAgent,
    questions: list[GeneratedQuestion],
    explore_references: list[dict[str, str]],
    verbose: bool = False,
) -> tuple[EvalMetrics, list[DetailedResult]]:
    """
    Run all questions against the system under test and collect metrics.
    """
    metrics = EvalMetrics()
    results = []

    total = len(questions)

    print(f"\n{'=' * 70}")
    print(f"  RUNNING EVALUATION — {total} questions")
    print(f"{'=' * 70}")

    for i, question in enumerate(questions, 1):
        if verbose:
            print(f"\n{'─' * 70}")
            print(f"  [{i}/{total}] [{question.difficulty.value}] {question.query}")
            print(f"  Category: {question.category} | Target: {question.target_model}.{question.target_explore}")

        # Call CA
        ca_response, latency_ms = await agent.call_ca_direct(
            question.query, explore_references,
        )

        # Record metrics
        metrics.record(question, ca_response, latency_ms)

        # Determine status
        if ca_response.has_error:
            status = "error"
        elif ca_response.answer and (ca_response.has_data or ca_response.has_query):
            status = "success"
        else:
            status = "failed"

        results.append(DetailedResult(
            question=question,
            ca_response=ca_response,
            latency_ms=latency_ms,
            status=status,
        ))

        # Print progress
        color = {"success": "\033[92m", "failed": "\033[93m", "error": "\033[91m"}
        reset = "\033[0m"
        if verbose:
            print(f"  Status: {color.get(status, '')}{status}{reset} ({latency_ms:.0f}ms)")
            if ca_response.answer:
                print(f"  Answer: {ca_response.answer[:150]}{'...' if len(ca_response.answer or '') > 150 else ''}")
            if ca_response.model_used:
                print(f"  Model: {ca_response.model_used}, Explore: {ca_response.explore_used}")
            if ca_response.has_error:
                print(f"  Error: {ca_response.error[:150]}")
        else:
            # Compact progress
            icon = {"success": "\033[92m✓\033[0m", "failed": "\033[93m✗\033[0m", "error": "\033[91m!\033[0m"}
            print(f"  {icon.get(status, '?')} [{question.difficulty.value[0].upper()}] {question.query[:60]}{'...' if len(question.query) > 60 else ''} ({latency_ms:.0f}ms)")

    return metrics, results


# ===========================================================================
# Output & Persistence
# ===========================================================================

def save_results(
    output_dir: Path,
    schema: LookerSchema,
    questions: list[GeneratedQuestion],
    metrics: EvalMetrics,
    results: list[DetailedResult],
):
    """Save all artifacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 1. Schema snapshot
    schema_path = output_dir / "schema_snapshot.json"
    with open(schema_path, "w") as f:
        json.dump(schema.to_dict(), f, indent=2)

    # 2. Generated questions
    questions_path = output_dir / "eval_questions.json"
    with open(questions_path, "w") as f:
        json.dump({
            "generated_at": timestamp,
            "schema_fingerprint": schema.instance_fingerprint,
            "count": len(questions),
            "questions": [q.to_dict() for q in questions],
        }, f, indent=2)

    # 3. Eval results (timestamped for trend tracking)
    results_path = output_dir / f"eval_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "schema_fingerprint": schema.instance_fingerprint,
            "system_under_test": "looker-conversational-analytics",
            "metrics": metrics.to_dict(),
            "results": [r.to_dict() for r in results],
        }, f, indent=2)

    # 4. Latest symlink (always points to most recent)
    latest_path = output_dir / "eval_results_latest.json"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    # Write a copy instead of symlink for portability
    with open(latest_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "schema_fingerprint": schema.instance_fingerprint,
            "system_under_test": "looker-conversational-analytics",
            "metrics": metrics.to_dict(),
            "results": [r.to_dict() for r in results],
        }, f, indent=2)

    print(f"\n  Artifacts saved to: {output_dir}/")
    print(f"    schema_snapshot.json     — Looker schema ({len(schema.explores)} explores)")
    print(f"    eval_questions.json      — {len(questions)} generated questions")
    print(f"    eval_results_{timestamp}.json — Full results + metrics")
    print(f"    eval_results_latest.json — Latest results (always current)")

    return results_path


# ===========================================================================
# Main Pipeline
# ===========================================================================

async def main():
    args = sys.argv[1:]
    verbose = "-v" in args or "--verbose" in args
    reuse_questions = "--reuse" in args
    generate_only = "--generate-only" in args
    count_per_difficulty = 5

    # Parse --count N
    if "--count" in args:
        idx = args.index("--count")
        if idx + 1 < len(args):
            count_per_difficulty = int(args[idx + 1])

    # Parse --questions <path>
    questions_file = None
    if "--questions" in args:
        idx = args.index("--questions")
        if idx + 1 < len(args):
            questions_file = Path(args[idx + 1])

    output_dir = Path(__file__).parent.parent / "eval_output"

    print("=" * 70)
    print("  Lumi Evaluation Pipeline")
    print("  Schema-Aware | Auto-Generated | Production-Grade")
    print("=" * 70)

    # --- Setup ---
    print("\n[1/6] Loading configuration...")
    config_path = Path(__file__).parent.parent / "config.yaml"
    env_path = Path(__file__).parent.parent / ".env"
    settings = load_settings(config_path=config_path, env_path=env_path)

    auth_client = IdaaSClient(settings.idaas)
    llm_provider = GeminiProvider(settings.llm, auth_client)

    print("[2/6] Connecting to MCP server...")
    looker_config = settings.mcp.servers.get("looker")
    if not looker_config:
        print("  Error: 'looker' MCP server not configured")
        return

    mcp_client = MCPClient(looker_config)
    try:
        await mcp_client.connect()
        print(f"  Connected — {len(mcp_client.tools)} tools available")
    except Exception as e:
        print(f"  Error: {e}")
        return

    agent = CAAgent(llm_provider, mcp_client)
    ca_tool = agent._find_ca_tool()
    if not ca_tool:
        print("  Warning: CA tool not found — eval will test manual query path")

    # --- Schema Discovery ---
    print("[3/6] Discovering Looker schema...")
    schema = await discover_schema(mcp_client, verbose=verbose)
    if not schema.explores:
        print("  No explores found. Cannot proceed.")
        await mcp_client.disconnect()
        return

    print(f"  Discovered: {len(schema.models)} models, {len(schema.explores)} explores, "
          f"{schema.total_dimensions} dims, {schema.total_measures} measures")

    # --- Question Generation ---
    questions = []

    if questions_file and questions_file.exists():
        print(f"[4/6] Loading questions from {questions_file}...")
        with open(questions_file) as f:
            data = json.load(f)
        questions = [GeneratedQuestion.from_dict(q) for q in data.get("questions", data if isinstance(data, list) else [])]
        print(f"  Loaded {len(questions)} questions")

    elif reuse_questions and (output_dir / "eval_questions.json").exists():
        print("[4/6] Reusing previously generated questions...")
        with open(output_dir / "eval_questions.json") as f:
            data = json.load(f)
        # Check if schema has changed
        if data.get("schema_fingerprint") != schema.instance_fingerprint:
            print(f"  Warning: Schema has changed since questions were generated!")
            print(f"    Old fingerprint: {data.get('schema_fingerprint')}")
            print(f"    New fingerprint: {schema.instance_fingerprint}")
            print(f"  Regenerating questions...")
            reuse_questions = False
        else:
            questions = [GeneratedQuestion.from_dict(q) for q in data["questions"]]
            print(f"  Loaded {len(questions)} questions (schema fingerprint matches)")

    if not questions:
        print(f"[4/6] Generating {count_per_difficulty} questions per difficulty level...")
        questions = await generate_questions(
            llm_provider, schema,
            count_per_difficulty=count_per_difficulty,
            verbose=verbose,
        )
        print(f"  Generated {len(questions)} questions total")

    if not questions:
        print("  No questions generated. Cannot proceed.")
        await mcp_client.disconnect()
        return

    # Print question summary
    by_diff = {}
    for q in questions:
        by_diff.setdefault(q.difficulty.value, []).append(q)
    print(f"\n  Question breakdown:")
    for diff, qs in by_diff.items():
        print(f"    {diff:14s}: {len(qs)} questions")
        if verbose:
            for q in qs[:3]:
                print(f"      • {q.query[:70]}{'...' if len(q.query) > 70 else ''}")
            if len(qs) > 3:
                print(f"      ... and {len(qs) - 3} more")

    if generate_only:
        # Save questions and exit
        output_dir.mkdir(parents=True, exist_ok=True)
        save_results(output_dir, schema, questions, EvalMetrics(), [])
        print("\n  Question generation complete. Use --reuse to run eval with these questions.")
        await mcp_client.disconnect()
        return

    # --- Build explore references ---
    explore_refs = [{"model": e.model, "explore": e.name} for e in schema.explores]
    # CA API accepts max 5 explore references
    if len(explore_refs) > 5:
        print(f"\n  Note: {len(explore_refs)} explores found, using first 5 for CA API")
        explore_refs = explore_refs[:5]

    # --- Run Evaluation ---
    print(f"[5/6] Running evaluation...")
    metrics, results = await run_evaluation(
        agent, questions, explore_refs, verbose=verbose,
    )

    # --- Report ---
    print("[6/6] Generating report...")
    metrics.print_report()

    # Save artifacts
    results_path = save_results(output_dir, schema, questions, metrics, results)

    # Target check
    print(f"\n  Target: 90%+ accuracy")
    if metrics.success_rate >= 90:
        print(f"  \033[92m  ✓ PASSING ({metrics.success_rate:.1f}%)\033[0m")
    elif metrics.success_rate >= 70:
        print(f"  \033[93m  ⚠ CLOSE ({metrics.success_rate:.1f}% — need {90 - metrics.success_rate:.1f}% more)\033[0m")
    else:
        print(f"  \033[91m  ✗ BELOW TARGET ({metrics.success_rate:.1f}%)\033[0m")

    await mcp_client.disconnect()
    print("\n[Done]")


if __name__ == "__main__":
    asyncio.run(main())
