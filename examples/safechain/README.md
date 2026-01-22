# SafeChain MCP Agent - ReAct Orchestration Layer

## Overview

This PoC demonstrates how to build an **intelligent, multi-step reasoning agent** using SafeChain's `MCPToolAgent` with a custom orchestration layer. The result is an interactive chatbot that can autonomously navigate complex data exploration workflows.

## The Problem

SafeChain provides `MCPToolAgent` - a powerful component that:
- Handles IdaaS authentication
- Manages LLM model access
- Executes MCP tools based on LLM decisions

However, `MCPToolAgent` performs **single-pass execution**:

```
User Query → LLM → Execute Tools → Return Results (done)
```

This works for simple queries but fails for complex workflows where the agent needs to:
1. Discover what data is available
2. Explore the data model
3. Build and execute a query
4. Analyze and present results

## The Solution

We built a **ReAct (Reasoning + Acting) orchestration layer** that wraps `MCPToolAgent`:

```
User Query
    ↓
┌─────────────────────────────────────────┐
│         SafeChainOrchestrator           │
│                                         │
│   ┌─────────────────┐                   │
│   │  MCPToolAgent   │ ← Single pass     │
│   └─────────────────┘                   │
│           ↓                             │
│   Tool Results?                         │
│       ├── YES → Feed back to LLM        │
│       │         ↓                       │
│       │   ┌─────────────────┐           │
│       │   │  MCPToolAgent   │ ← Again   │
│       │   └─────────────────┘           │
│       │         ↓                       │
│       │   (repeat until done)           │
│       │                                 │
│       └── NO → Return Final Answer      │
│                                         │
└─────────────────────────────────────────┘
```

## What This Unlocks

### 1. Autonomous Multi-Step Workflows

A generic query like **"Show me total sales by region"** triggers an autonomous workflow:

| Step | Agent Action | Tool Used |
|------|--------------|-----------|
| 1 | "I need to find available models" | `get_models()` |
| 2 | "sales_analytics looks relevant" | `get_explores(model="sales_analytics")` |
| 3 | "regional_sales has what I need" | `get_dimensions()`, `get_measures()` |
| 4 | "Now I can build the query" | `query(dimensions=[region], measures=[total_sales])` |
| 5 | "Here are the results..." | Final answer (no tool call) |

The user doesn't need to know the data model - the agent discovers it autonomously.

### 2. Real-Time Thinking Visualization

Watch the agent's reasoning process as it works:

```
╭──────────────────── Tool Call: get_models ────────────────────╮
│ Executing: get_models                                         │
╰───────────────────────────────────────────────────────────────╯

╭──────────────────── Tool Result ──────────────────────────────╮
│ [{"name": "ecommerce", "label": "E-Commerce"},                │
│  {"name": "sales_analytics", "label": "Sales Analytics"}]     │
╰───────────────────────────────────────────────────────────────╯

╭──────────────────── Thinking ─────────────────────────────────╮
│ I found two models. For sales by region, I should explore     │
│ the sales_analytics model...                                  │
╰───────────────────────────────────────────────────────────────╯
```

### 3. Conversation Memory

The agent remembers context across turns:

```
You: What models are available?
Agent: I found ecommerce and sales_analytics models...

You: Tell me more about the second one
Agent: The sales_analytics model has these explores... (knows "second one" = sales_analytics)
```

### 4. Error Recovery

If a tool fails, the agent sees the error and can try alternative approaches:

```
Tool Error: Explore 'sales' not found
Agent: "Let me check the available explores again..."
       → Calls get_explores() to find correct name
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        chat.py                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │   ChatSession   │    │    SafeChainOrchestrator         │   │
│  │                 │    │                                  │   │
│  │ - History mgmt  │───▶│ - ReAct loop                     │   │
│  │ - CLI commands  │    │ - Message formatting             │   │
│  │                 │    │ - Thinking events                │   │
│  └─────────────────┘    │                                  │   │
│                         │    ┌────────────────────────┐    │   │
│                         │    │    MCPToolAgent        │    │   │
│                         │    │    (from safechain)    │    │   │
│                         │    │                        │    │   │
│                         │    │ - IdaaS auth           │    │   │
│                         │    │ - LLM access           │    │   │
│                         │    │ - Tool execution       │    │   │
│                         │    └────────────────────────┘    │   │
│                         └──────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              ConsoleThinkingCallback                     │   │
│  │                                                          │   │
│  │  Real-time visualization with rich formatting            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SafeChain Library                           │
├─────────────────────────────────────────────────────────────────┤
│  MCPToolLoader    - Loads tools from MCP servers                │
│  MCPToolAgent     - LLM + tool execution                        │
│  IdaaS Auth       - Enterprise authentication                   │
│  Model Access     - Gemini/LLM API access                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Servers                                │
├─────────────────────────────────────────────────────────────────┤
│  Looker MCP       - Data exploration & querying                 │
│  Other MCPs       - Additional capabilities                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### SafeChainOrchestrator

The core orchestration layer that implements the ReAct pattern:

```python
orchestrator = SafeChainOrchestrator(
    model_id="gemini-pro",           # Model identifier
    tools=tools,                      # From MCPToolLoader
    system_prompt=SYSTEM_PROMPT,      # Guides agent behavior
    max_iterations=15,                # Prevents infinite loops
    thinking_callback=callback,       # Real-time visualization
)

result = await orchestrator.run(messages)
# result = {"content": "Final answer...", "thinking_events": [...]}
```

### ThinkingCallback

Pluggable visualization system:

```python
class ConsoleThinkingCallback(ThinkingCallback):
    def on_thinking(self, event: ThinkingEvent):
        # Display reasoning, tool calls, results in real-time
        ...

# Can be extended for:
# - Web UI streaming
# - Logging systems
# - Custom displays
```

### ChatSession

Conversation management:

```python
session = ChatSession(orchestrator)
response = await session.chat("Show me sales data")
# Automatically manages history, context window
```

## Usage

```bash
# Ensure .env has required credentials
python examples/safechain/chat.py
```

### Commands

| Command | Description |
|---------|-------------|
| `/tools` | List available MCP tools |
| `/clear` | Clear conversation history |
| `/help` | Show help |
| `/quit` | Exit |

### Example Session

```
You: What data can I explore?

╭─────────── Tool Call: get_models ───────────╮
│ Executing: get_models                       │
╰─────────────────────────────────────────────╯

╭─────────── Tool Result ─────────────────────╮
│ [{name: "ecommerce"}, {name: "sales"}]      │
╰─────────────────────────────────────────────╯

╭─────────── Answer ──────────────────────────╮
│ You have access to 2 models:                │
│ 1. **ecommerce** - E-commerce transactions  │
│ 2. **sales** - Sales analytics              │
╰─────────────────────────────────────────────╯

────────────────────────────────────────────────
Assistant: You have access to 2 models...
────────────────────────────────────────────────

You: Show me revenue by category from ecommerce

╭─────────── Tool Call: get_explores ─────────╮
│ Executing: get_explores                     │
╰─────────────────────────────────────────────╯

╭─────────── Tool Result ─────────────────────╮
│ [order_items, products, users]              │
╰─────────────────────────────────────────────╯

╭─────────── Thinking ────────────────────────╮
│ order_items likely has revenue data...      │
╰─────────────────────────────────────────────╯

╭─────────── Tool Call: get_measures ─────────╮
│ Executing: get_measures                     │
╰─────────────────────────────────────────────╯

... (agent continues discovering, then queries)

╭─────────── Tool Call: query ────────────────╮
│ Executing: query                            │
╰─────────────────────────────────────────────╯

╭─────────── Answer ──────────────────────────╮
│ Here's the revenue by category:             │
│                                             │
│ | Category    | Revenue    |                │
│ |-------------|------------|                │
│ | Electronics | $1,245,000 |                │
│ | Clothing    | $892,000   |                │
│ | Home        | $654,000   |                │
╰─────────────────────────────────────────────╯
```

## Configuration

The system uses SafeChain's configuration via `ee_config`:

```python
from ee_config.config import Config
config = Config.from_env()  # Reads from .env file
```

Required environment variables (managed by SafeChain/ee_config):
- IdaaS credentials
- Model configuration
- MCP server endpoints

## Extending the System

### Custom Thinking Callback

```python
class WebSocketThinkingCallback(ThinkingCallback):
    def __init__(self, websocket):
        self.ws = websocket

    def on_thinking(self, event: ThinkingEvent):
        # Stream to web UI
        asyncio.create_task(self.ws.send_json({
            "type": event.type.value,
            "content": event.content,
        }))
```

### Custom System Prompt

Modify `SYSTEM_PROMPT` in `chat.py` to change agent behavior:

```python
SYSTEM_PROMPT = """You are a financial analyst assistant...

When asked about financial data:
1. Always check compliance requirements first
2. Verify user has access to requested data
3. ...
"""
```

### Programmatic Usage

```python
from examples.safechain import SafeChainOrchestrator, ConsoleThinkingCallback

async def analyze_data(query: str):
    config = Config.from_env()
    tools = await MCPToolLoader.load_tools(config)

    orchestrator = SafeChainOrchestrator(
        model_id="gemini-pro",
        tools=tools,
        max_iterations=10,
    )

    result = await orchestrator.run([
        {"role": "user", "content": query}
    ])

    return result["content"]
```

## Summary

This PoC demonstrates that by adding a thin orchestration layer (~150 lines) around SafeChain's `MCPToolAgent`, we can transform a single-pass tool executor into a fully autonomous reasoning agent capable of:

- **Multi-step discovery and exploration**
- **Dynamic decision making based on intermediate results**
- **Real-time thinking visualization**
- **Conversation memory and context management**
- **Error recovery and alternative approaches**

The user simply asks a question in natural language, and the agent autonomously figures out how to answer it using the available tools.