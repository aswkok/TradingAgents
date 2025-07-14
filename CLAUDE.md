# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TradingAgents is a multi-agent LLM financial trading framework that simulates the structure of real-world trading firms. The system employs specialized agents across five teams:

1. **Analyst Team** - Performs specialized analysis (market, social sentiment, news, fundamentals)
2. **Researcher Team** - Conducts bull/bear debates and creates investment plans
3. **Trader Agent** - Makes trading decisions based on research
4. **Risk Management Team** - Evaluates risk through multi-perspective analysis
5. **Portfolio Manager** - Makes final trading decisions

## Core Architecture

### Key Components

- **TradingAgentsGraph** (`tradingagents/graph/trading_graph.py`): Main orchestration class that coordinates all agents using LangGraph
- **Agent States** (`tradingagents/agents/utils/agent_states.py`): Defines the state structures for different debate phases (InvestDebateState, RiskDebateState, AgentState)
- **Memory System** (`tradingagents/agents/utils/memory.py`): FinancialSituationMemory for each agent to learn from past decisions
- **Configuration** (`tradingagents/default_config.py`): Central configuration with LLM settings, debate rounds, and tool preferences

### Data Flow Architecture

The system follows a sequential pipeline:
1. **Data Collection**: Agents use specialized tools to gather market data, news, sentiment, and fundamentals
2. **Analysis**: Each analyst produces specialized reports
3. **Debate**: Bull/bear researchers debate investment merits
4. **Trading**: Trader creates investment plan based on research
5. **Risk Assessment**: Risk management team evaluates through multi-perspective debate
6. **Final Decision**: Portfolio manager makes final trading decision

### LLM Integration

- Supports OpenAI, Anthropic, Google, and other providers via LangChain
- Uses two-tier LLM system: deep_think_llm for complex analysis, quick_think_llm for rapid decisions
- Configurable through `config["llm_provider"]`, `config["deep_think_llm"]`, `config["quick_think_llm"]`

## Environment Setup

### Dependencies Installation
```bash
# Create conda environment
conda create -n tradingagents python=3.13
conda activate tradingagents

# Install dependencies
pip install -r requirements.txt
```

### Required API Keys
```bash
export FINNHUB_API_KEY=your_finnhub_api_key
export OPENAI_API_KEY=your_openai_api_key
```

## Development Commands

### Running the System

**CLI Interface:**
```bash
python -m cli.main
```

**Direct Python Usage:**
```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())
_, decision = ta.propagate("NVDA", "2024-05-10")
```

### Configuration Customization

Modify `DEFAULT_CONFIG` for different LLM providers, debate rounds, or data sources:

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "anthropic"  # or "google", "openai"
config["deep_think_llm"] = "claude-3-opus-20240229"
config["max_debate_rounds"] = 2
config["online_tools"] = True  # Use real-time data vs cached
```

## Key Development Patterns

### Agent Structure
- Each agent inherits from base classes in `tradingagents/agents/`
- Agents use LangChain tools for data access via `tradingagents/dataflows/`
- Memory persistence through `FinancialSituationMemory` for learning from past decisions

### State Management
- `AgentState` tracks the complete pipeline state
- `InvestDebateState` and `RiskDebateState` manage debate phases
- State flows through LangGraph nodes with conditional logic

### Tool Integration
- Online tools: Real-time data from APIs (FINNHUB, Yahoo Finance, Google News)
- Offline tools: Cached data from TauricDB for backtesting
- Tool selection via `config["online_tools"]` boolean

### Debate System
- Bull/bear researchers engage in structured debates
- Risk management team uses three perspectives (risky, safe, neutral)
- Debate rounds controlled by `max_debate_rounds` and `max_risk_discuss_rounds`

## File Structure Context

- `cli/`: Rich-based CLI interface with real-time progress display
- `tradingagents/agents/`: Individual agent implementations organized by team
- `tradingagents/dataflows/`: Data source integrations and utilities
- `tradingagents/graph/`: LangGraph orchestration and workflow logic
- `main.py`: Simple example script for direct usage
- `setup.py`: Package setup configuration

## Testing and Validation

No formal test suite exists. Validation typically involves:
- Running complete trading scenarios with known market dates
- Comparing online vs offline tool results
- Evaluating agent debate quality and final decisions
- Memory system effectiveness through backtesting

## Important Notes

- Framework is research-oriented, not production trading advice
- Heavy API usage - consider cost implications with `o1-preview` models
- Recommend `o4-mini` and `gpt-4.1-mini` for testing to reduce costs
- Memory system allows agents to learn from past trading performance
- Debug mode (`debug=True`) provides detailed execution tracing