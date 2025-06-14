# Agentic Ecommerce Chatbot Demo

Simple demo of an AI shopping assistant using LangGraph and Graph Neural Networks.

## Prerequisites

- Docker Desktop running
- VS Code with Dev Containers extension
- Anthropic API key

## Quick Start

1. Open the project in VS Code
2. Click "Reopen in Container" when prompted
3. Wait for container to build (~2 minutes)

## Running the Demo

Once inside the container, open two terminals:

**Terminal 1 - Start the API:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
python -m app.main
```

**Terminal 2 - Run the chat:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
python -m cli chat
```

## Sample Conversation

Try these messages:
- "Hi, I need clothes for work"
- "I like business casual style"
- "Navy and white colors, budget $80"

## Other Commands

```bash
# Run tests
python -m pytest tests/

# View all products
python -m cli products

# Check graph stats
python -m cli stats

# Run scripted demo
python -m cli demo
```

## Troubleshooting

If Docker fails, restart Docker Desktop and try again.

If API won't start, make sure port 8000 is free.

If "API key not found", check you exported ANTHROPIC_API_KEY correctly.
