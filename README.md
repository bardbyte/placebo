# Lumi LLM

A reusable Python package for enterprise LLM access with MCP (Model Context Protocol) tool binding. Built to demonstrate natural language to SQL generation using Google's Looker MCP Toolbox and Gemini.

## Overview

Lumi LLM provides a clean abstraction layer for:
- **Enterprise LLM Access**: Connect to LLMs (Gemini, OpenAI, Azure OpenAI) through internal proxies with IdaaS authentication
- **MCP Tool Integration**: Bind tools from MCP servers (like Looker Toolbox) to LLMs for agentic workflows
- **Transparent Reasoning**: LangGraph-based agent that shows its thinking process step-by-step

### Use Case Demo

Ask natural language questions about your data, and the agent will:
1. Discover available data models in Looker
2. Explore dimensions and measures
3. Generate and execute SQL queries
4. Present results with explanations

```
You: Show me total sales by region for last month

╭─ Thinking ─────────────────────────────────────────────────╮
│ I need to first discover what data models are available... │
╰────────────────────────────────────────────────────────────╯

╭─ Tool Call: get-models ────────────────────────────────────╮
│ Calling tool with arguments: {}                            │
╰────────────────────────────────────────────────────────────╯

╭─ Tool Result ──────────────────────────────────────────────╮
│ ["ecommerce", "marketing", "finance"]                      │
╰────────────────────────────────────────────────────────────╯

... (agent continues exploring and querying)

╭─ Answer ───────────────────────────────────────────────────╮
│ Here's the SQL and results for total sales by region:      │
│                                                            │
│ SELECT region, SUM(sale_amount) as total_sales             │
│ FROM order_items                                           │
│ WHERE order_date >= '2024-12-01'                           │
│ GROUP BY region                                            │
│                                                            │
│ Results:                                                   │
│ | Region | Total Sales |                                   │
│ |--------|-------------|                                   │
│ | West   | $1,234,567  |                                   │
│ | East   | $987,654    |                                   │
╰────────────────────────────────────────────────────────────╯
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Your Machine                               │
│                                                                         │
│  ┌───────────────────────────────┐    ┌───────────────────────────────┐ │
│  │         Terminal 1            │    │         Terminal 2            │ │
│  │                               │    │                               │ │
│  │   MCP Toolbox Server          │    │   Python Application          │ │
│  │   (./toolbox)                 │    │   (python examples/chat.py)   │ │
│  │                               │    │                               │ │
│  │   Reads: tools.yaml           │    │   Reads: config.yaml, .env    │ │
│  │   Exposes: Looker tools       │    │   Uses: Gemini + MCP tools    │ │
│  │   Port: 5000                  │    │                               │ │
│  └───────────────┬───────────────┘    └───────────────┬───────────────┘ │
│                  │                                    │                 │
│                  │◄───────── SSE Connection ─────────►│                 │
│                  │         (localhost:5000/mcp)       │                 │
└──────────────────┼────────────────────────────────────┼─────────────────┘
                   │                                    │
                   │ HTTPS                              │ HTTPS
                   │ (Looker API)                       │ (IdaaS + Gemini)
                   ▼                                    ▼
        ┌─────────────────────┐              ┌─────────────────────┐
        │   Looker Instance   │              │   Internal Proxy    │
        │                     │              │                     │
        │  • LookML Models    │              │  • IdaaS Auth       │
        │  • Explores         │              │  • Gemini API       │
        │  • Dimensions       │              │  • OpenAI API       │
        │  • Measures         │              │  • Azure OpenAI     │
        └─────────────────────┘              └─────────────────────┘
```

## Features

| Feature | Description |
|---------|-------------|
| **IdaaS Authentication** | Token acquisition with automatic refresh and caching |
| **Multi-Provider Support** | Gemini, OpenAI, Azure OpenAI through internal proxies |
| **MCP Client** | SSE-based client for Model Context Protocol servers |
| **Tool Binding** | Automatic conversion of MCP tools to LLM function calling format |
| **LangGraph Agent** | ReAct-style agent with configurable tool call limits |
| **Thinking Callbacks** | Real-time display of agent reasoning (console or custom) |
| **Async & Sync APIs** | Both async and synchronous interfaces throughout |

## Prerequisites

- **Python 3.11+**
- **Looker Instance** with API access
- **IdaaS Credentials** (CIBIS) for LLM proxy access
- **MCP Toolbox** binary (downloaded in setup)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bardbyte/placebo.git
cd placebo
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download MCP Toolbox

**macOS (Apple Silicon):**
```bash
export VERSION=0.24.0
curl -L -o toolbox https://storage.googleapis.com/genai-toolbox/v$VERSION/darwin/arm64/toolbox
chmod +x toolbox
```

**macOS (Intel):**
```bash
export VERSION=0.24.0
curl -L -o toolbox https://storage.googleapis.com/genai-toolbox/v$VERSION/darwin/amd64/toolbox
chmod +x toolbox
```

**Linux (AMD64):**
```bash
export VERSION=0.24.0
curl -L -o toolbox https://storage.googleapis.com/genai-toolbox/v$VERSION/linux/amd64/toolbox
chmod +x toolbox
```

**Windows (PowerShell):**
```powershell
$VERSION = "0.24.0"
curl.exe -o toolbox.exe "https://storage.googleapis.com/genai-toolbox/v$VERSION/windows/amd64/toolbox.exe"
```

## Configuration

### 1. Environment Variables (`.env`)

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# IdaaS Authentication (for LLM proxy)
CIBIS_CONSUMER_INTEGRATION_ID=your_integration_id
CIBIS_CONSUMER_SECRET=your_secret

# Looker Configuration (for MCP Toolbox)
LOOKER_INSTANCE_URL=https://your-company.looker.com
LOOKER_CLIENT_ID=your_looker_api_client_id
LOOKER_CLIENT_SECRET=your_looker_api_client_secret
```

**Getting Looker API Credentials:**
1. Log into your Looker instance
2. Go to **Admin** → **Users**
3. Find your user and click **Edit**
4. Scroll to **API Keys** and click **New API Key**
5. Copy the Client ID and Client Secret

### 2. Application Config (`config.yaml`)

This file configures the Python application. The default configuration should work for most cases:

```yaml
idaas:
  id: ${CIBIS_CONSUMER_INTEGRATION_ID}
  url: "https://eag-dev.aexp.com/oneidentity/security/digital/v1/application/token"
  secret: ${CIBIS_CONSUMER_SECRET}
  token_refresh_time: 60

llm:
  default_provider: "gemini"
  providers:
    gemini:
      url: "https://eag-dev.aexp.com/genai/google/v1/models/gemini-1.5-pro-002/generateContent"
      scope: ["/genai/google/v1/models/gemini-1.5-pro-002/#*::post"]
      temperature: 0.5
      max_tokens: 1024

mcp:
  servers:
    looker:
      url: "http://localhost:5000/mcp"
      transport: "sse"
```

### 3. MCP Toolbox Config (`tools.yaml`)

This file configures the MCP Toolbox server. It defines:
- **Source**: Connection to your Looker instance
- **Tools**: Which Looker tools to expose
- **Toolsets**: Groupings of tools

The default configuration includes 16 essential tools for data exploration:

| Category | Tools |
|----------|-------|
| Data Model Discovery | `get-models`, `get-explores`, `get-dimensions`, `get-measures`, `get-filters`, `get-parameters` |
| Query Execution | `query`, `query-sql`, `query-url` |
| Saved Content | `get-looks`, `run-look`, `get-dashboards`, `run-dashboard` |
| LookML Projects | `get-projects`, `get-project-files`, `get-project-file` |

## Running the Application

### Step 1: Start MCP Toolbox Server

In **Terminal 1**:

```bash
cd placebo

# Load environment variables
source .env
export LOOKER_INSTANCE_URL LOOKER_CLIENT_ID LOOKER_CLIENT_SECRET

# Start the MCP server
./toolbox --tools-file tools.yaml
```

You should see:
```
2024/01/20 10:00:00 Starting MCP Toolbox server...
2024/01/20 10:00:00 Loaded source: my-looker (looker)
2024/01/20 10:00:00 Loaded 16 tools
2024/01/20 10:00:00 Server listening on :5000
```

Keep this terminal running.

### Step 2: Run the Interactive Chat

In **Terminal 2**:

```bash
cd placebo
source venv/bin/activate

# Run the interactive chat
python examples/chat.py
```

### Step 3: Start Chatting

```
╔═════════════════════════════════════════════════════════════╗
║              LUMI LLM - Interactive Chat                    ║
║         Natural Language to SQL with Looker MCP             ║
╚═════════════════════════════════════════════════════════════╝

[1/4] Loading configuration...
[2/4] Initializing authentication...
[3/4] Initializing Gemini provider...
[4/4] Connecting to MCP server...
      Connected! Found 16 tools

============================================================
AVAILABLE TOOLS
============================================================

Models & Explores:
  - get-models
  - get-explores

Fields:
  - get-dimensions
  - get-measures
  ...

Type your question or command. Use /quit to exit.

You: What LookML projects are available?
```

### Chat Commands

| Command | Description |
|---------|-------------|
| `/tools` | List all available MCP tools |
| `/clear` | Clear conversation history |
| `/help` | Show help and example queries |
| `/quit` | Exit the chat |

### Example Queries

```
"What tools do you have access to?"
"What LookML projects are available?"
"Show me the files in the ecommerce project"
"What models are available?"
"What explores are in the ecommerce model?"
"Show me dimensions in the order_items explore"
"Show me total sales by region for last month"
```

## Project Structure

```
placebo/
├── config.yaml              # Python app configuration
├── tools.yaml               # MCP Toolbox configuration
├── .env.example             # Environment variables template
├── .env                     # Your credentials (git-ignored)
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Package configuration
├── toolbox                  # MCP Toolbox binary (git-ignored)
│
├── lumi_llm/                # Main Python package
│   ├── __init__.py
│   │
│   ├── config/              # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py      # Pydantic settings with env var substitution
│   │
│   ├── auth/                # Authentication
│   │   ├── __init__.py
│   │   └── idaas.py         # IdaaS token acquisition and caching
│   │
│   ├── providers/           # LLM Providers
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract provider interface
│   │   └── gemini.py        # Gemini implementation
│   │
│   ├── mcp/                 # MCP Client
│   │   ├── __init__.py
│   │   ├── client.py        # SSE-based MCP client
│   │   └── tool_converter.py # MCP → LLM tool format conversion
│   │
│   └── agents/              # LangGraph Agents
│       ├── __init__.py
│       └── tool_agent.py    # ReAct agent with thinking callbacks
│
└── examples/                # Demo scripts
    ├── __init__.py
    ├── chat.py              # Interactive CLI chatbot
    ├── looker_nl_to_sql.py  # Single-query demo
    └── test_setup.py        # Setup verification script
```

## How It Works

### Agent Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                          │
│                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│  │  START  │───►│   LLM   │───►│  Tools  │───►│   LLM   │  │
│  └─────────┘    └────┬────┘    └────┬────┘    └────┬────┘  │
│                      │              │              │        │
│                      │   Tool       │   Tool       │        │
│                      │   Calls?     │   Results    │        │
│                      │              │              │        │
│                      ▼              ▼              ▼        │
│                 [Thinking]    [Thinking]     [Thinking]     │
│                  Callback      Callback       Callback      │
│                                                             │
│                                         No more tools?      │
│                                              │              │
│                                              ▼              │
│                                        ┌─────────┐          │
│                                        │ Finalize│          │
│                                        └────┬────┘          │
│                                             │               │
└─────────────────────────────────────────────┼───────────────┘
                                              │
                                              ▼
                                        Final Answer
```

### Component Interaction

1. **User Query** → Chat interface receives natural language question
2. **LangGraph Agent** → Orchestrates the reasoning loop
3. **Gemini Provider** → Sends messages to LLM, receives tool calls
4. **MCP Client** → Executes tools via MCP Toolbox
5. **Looker** → Provides data model info and query results
6. **Thinking Callback** → Displays each step to the user

## API Reference

### Quick Start

```python
import asyncio
from lumi_llm import (
    load_settings,
    IdaaSClient,
    GeminiProvider,
    MCPClient,
    create_tool_agent,
    ConsoleThinkingCallback,
)

async def main():
    # Load configuration
    settings = load_settings()

    # Initialize components
    auth = IdaaSClient(settings.idaas)
    llm = GeminiProvider(settings.llm.providers["gemini"], auth)
    mcp = MCPClient(settings.mcp.servers["looker"])

    # Connect to MCP server
    await mcp.connect()

    # Create agent
    callback = ConsoleThinkingCallback()
    agent = create_tool_agent(llm, mcp, thinking_callback=callback)

    # Run query
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What models are available?"}],
        "thinking_events": [],
        "tool_calls_made": 0,
        "final_answer": None,
    })

    print(result["final_answer"])

asyncio.run(main())
```

### Custom Thinking Callback

```python
from lumi_llm.agents import ThinkingCallback, ThinkingEvent, ThinkingType

class MyCallback(ThinkingCallback):
    def on_thinking(self, event: ThinkingEvent) -> None:
        if event.type == ThinkingType.TOOL_CALL:
            print(f"Calling: {event.metadata['tool_name']}")
        elif event.type == ThinkingType.FINAL_ANSWER:
            print(f"Answer: {event.content}")
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `Connection refused localhost:5000` | MCP Toolbox not running | Start `./toolbox --tools-file tools.yaml` in Terminal 1 |
| `401 Unauthorized` from IdaaS | Invalid CIBIS credentials | Check `CIBIS_CONSUMER_INTEGRATION_ID` and `CIBIS_CONSUMER_SECRET` in `.env` |
| `Looker API error` | Invalid Looker credentials | Verify `LOOKER_*` variables and API key permissions |
| `No tools found` | tools.yaml misconfigured | Check YAML syntax, verify toolbox loaded tools |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `Token expired` | IdaaS token timeout | Token auto-refreshes; if persistent, check `token_refresh_time` |

### Verify Setup

Run the setup verification script:

```bash
python examples/test_setup.py
```

This checks:
- All imports work correctly
- Configuration files are valid
- Thinking callback system functions

## Future Enhancements

- [ ] OpenAI and Azure OpenAI provider implementations
- [ ] Streaming responses in the agent
- [ ] BigQuery MCP integration
- [ ] Web UI for the chat interface
- [ ] Conversation persistence
- [ ] Rate limiting and retry logic
- [ ] Comprehensive test suite

## References

- [MCP Toolbox for Databases](https://googleapis.github.io/genai-toolbox/)
- [Looker MCP Tools Documentation](https://googleapis.github.io/genai-toolbox/resources/tools/looker/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)

## License

MIT License - See [LICENSE](LICENSE) for details.
