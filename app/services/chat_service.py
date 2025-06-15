"""Chat service for orchestrating conversations."""

from typing import List, Dict, Any, Optional
from langchain_anthropic import ChatAnthropic
from app.agents.ecommerce_agent import EcommerceAgent
from app.gnn.engine import RecommendationEngine
from app.models.api_models import ChatMessage, ChatRequest, ChatResponse, ProductRecommendation
from app.services.graph_service import GraphService
import os


class ChatService:
    """Service for managing chat conversations."""
    
    def __init__(self, graph_service: GraphService):
        self.graph_service = graph_service
        self.recommendation_engine = RecommendationEngine()
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model=os.getenv("LLM_MODEL", "claude-3-opus-20240229"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "500"))
        )
        
        # Initialize agent
        self.agent = EcommerceAgent(
            llm=self.llm,
            graph_service=self.graph_service,
            recommendation_engine=self.recommendation_engine
        )
        
        # Give agent reference to chat service for history access
        self.agent.chat_service = self
        
        # Session storage (in-memory for demo)
        self.sessions: Dict[str, List[ChatMessage]] = {}
    
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request and return response."""
        # Get or create session
        session_id = request.session_id or self._generate_session_id()
        
        # Get or create user in graph
        user = self.graph_service.get_or_create_user(session_id)
        
        # Get conversation history
        history = self.sessions.get(session_id, [])
        
        # Add new message to history
        history.append(ChatMessage(role="user", content=request.message))
        
        # Process with agent - now pass the full history
        response_text, recommendations, follow_up = await self.agent.process_message(
            message=request.message,
            session_id=session_id,
            conversation_history=history[:-1]  # All previous messages (exclude current)
        )
        
        # Add response to history
        history.append(ChatMessage(role="assistant", content=response_text))
        
        # Update session history
        self.sessions[session_id] = history[-10:]  # Keep last 10 messages
        
        return ChatResponse(
            message=response_text,
            recommendations=recommendations,
            session_id=session_id,
            follow_up_question=follow_up
        )
    
    def get_session_history(self, session_id: str) -> List[ChatMessage]:
        """Get chat history for a session."""
        return self.sessions.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """Clear a session's history."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def _generate_session_id(self) -> str:
        """Generate a new session ID."""
        import uuid
        return str(uuid.uuid4())[:8]