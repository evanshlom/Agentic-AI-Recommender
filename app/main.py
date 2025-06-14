"""Main FastAPI application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.api import chat, products, recommendations
from app.services.graph_service import GraphService
from app.services.chat_service import ChatService
from app.models.api_models import HealthResponse

# Load environment variables
load_dotenv()

# Global services
graph_service = None
chat_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global graph_service, chat_service
    
    # Initialize services
    graph_service = GraphService()
    chat_service = ChatService(graph_service)
    
    # Set services in app state
    app.state.graph_service = graph_service
    app.state.chat_service = chat_service
    
    yield
    
    # Cleanup (if needed)


# Create FastAPI app
app = FastAPI(
    title="Agentic Ecommerce Chatbot API",
    description="AI-powered ecommerce assistant with graph-based recommendations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(products.router, prefix="/api/products", tags=["products"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint."""
    return HealthResponse(
        status="healthy",
        services={
            "graph": "initialized",
            "chat": "ready",
            "recommendations": "ready"
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        services={
            "graph": "active",
            "chat": "active",
            "recommendations": "active"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )