# Agentic AI Recommender System

**AI-powered ecommerce chatbot using LangGraph, Graph Neural Networks, and FastAPI**

## Features

- 🧠 **Graph Neural Networks** for product recommendations
- 🤖 **LangGraph agents** for conversational AI
- 🌐 **FastAPI** REST API with interactive docs
- 💬 **CLI chat interface** for testing
- 📊 **Product catalog** with filtering and search
- ✅ **Comprehensive test suite** (20 tests covering GNN, Agent, API)

## Prerequisites

- Docker Desktop
- VS Code with Dev Containers extension
- Anthropic API key

## Setup & Installation

### 1. Clone and Open in Dev Container

```bash
git clone <your-repo>
cd Agentic-AI-Recommender
```

Open in VS Code and click **"Reopen in Container"** when prompted.

### 2. Verify Installation

Once the container builds, verify everything works:

```bash
# Run all tests (should pass 20/20)
python -m pytest -v

# Expected: 20 passed, 7 warnings
```

## Usage

### Start the API Server

**Terminal 1:**
```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-ant-your-key-here"

# Start the FastAPI server
python -m app.main
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Demo Commands

**Terminal 2 (while server is running):**

```bash
# View all products in nice table format
python -m cli products

# Start interactive chat session
python -m cli chat

# Filter products
python -m cli products --style casual
python -m cli products --category shirts --max-price 80

# Check system health
curl http://localhost:8000/health
```

### Interactive Chat Demo

```bash
python -m cli chat
```

Try these conversations:
- *"Hi, I need clothes for work"*
- *"I want casual shirts for the weekend"*
- *"Something in blue under $70"*
- *"What about matching pants?"*

### API Endpoints

```bash
# Product catalog
curl http://localhost:8000/api/products/

# Chat API
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "I need business attire", "session_id": "demo"}'

# Graph statistics
curl http://localhost:8000/api/recommendations/graph/stats

# Interactive API docs
open http://localhost:8000/docs
```

## Testing

```bash
# Run all tests
python -m pytest -v

# Run specific test suites
python -m pytest tests/test_gnn.py -v      # GNN models (8 tests)
python -m pytest tests/test_agent.py -v   # LangGraph agent (5 tests)  
python -m pytest tests/test_api.py -v     # FastAPI endpoints (7 tests)

# Run with coverage
python -m pytest --cov=app
```

## Project Structure

```
├── app/
│   ├── agents/          # LangGraph conversation agents
│   ├── api/             # FastAPI REST endpoints
│   ├── gnn/             # Graph Neural Network models
│   ├── models/          # Pydantic data models
│   └── services/        # Business logic services
├── cli/                 # Command-line interface
├── tests/               # Comprehensive test suite
├── requirements.txt     # Python dependencies
└── .devcontainer/       # Docker dev environment
```

## System Architecture

1. **User Input** → CLI or API endpoint
2. **LangGraph Agent** → Processes conversation and extracts preferences  
3. **Graph Neural Network** → Computes product embeddings and similarities
4. **Recommendation Engine** → Generates scored product recommendations
5. **Response** → Formatted recommendations with explanations

## Known Issues

- ⚠️ **Search functionality needs improvement**: Product search by name/keywords not working optimally
- ⚠️ **Recommendation scoring**: All products receiving similar scores instead of contextual ranking
- ℹ️ **PyTorch warnings**: Expected warnings about user node updates (doesn't affect functionality)

## Development Status

✅ **Core Infrastructure**: All tests passing, Docker environment working  
✅ **API Endpoints**: FastAPI server with full CRUD operations  
✅ **GNN Models**: Graph neural networks training and generating embeddings  
✅ **Conversation AI**: LangGraph agents processing natural language  
🔧 **Recommendation Logic**: Needs refinement for better product matching  

## Troubleshooting

**Tests failing?**
```bash
# Rebuild container
# Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

**API won't start?**
```bash
# Check if port 8000 is free
lsof -i :8000
```

**Connection failed?**
```bash
# Verify server is running first
curl http://localhost:8000/health
```

**Chat not working?**
```bash
# Make sure API key is set
echo $ANTHROPIC_API_KEY
```

## Contributing

1. All changes should pass the test suite: `python -m pytest -v`
2. Follow the existing code style and patterns
3. Add tests for new features
4. Update this README for significant changes

## License

[Your License Here]