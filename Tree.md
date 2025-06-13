agentic-ecommerce-chatbot/
│   .env.example
│   README.md
│   requirements.txt
│
├───app/
│   │   __init__.py
│   │   main.py
│   │
│   ├───agents/
│   │       __init__.py
│   │       ecommerce_agent.py
│   │       prompts.py
│   │       state.py
│   │
│   ├───api/
│   │       __init__.py
│   │       chat.py
│   │       products.py
│   │       recommendations.py
│   │
│   ├───gnn/
│   │       __init__.py
│   │       engine.py
│   │       models.py
│   │
│   ├───models/
│   │       __init__.py
│   │       api_models.py
│   │       graph_models.py
│   │
│   └───services/
│           __init__.py
│           chat_service.py
│           graph_service.py
│
├───cli/
│       __init__.py
│       chat.py
│       client.py
│       main.py
│
└───tests/
        __init__.py
        test_agent.py
        test_api.py
        test_gnn.py