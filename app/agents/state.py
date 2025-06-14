"""State management for the LangGraph agent."""

from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import MessagesState
from app.models.api_models import ChatMessage, ProductRecommendation


class ConversationState(TypedDict):
    """State for the ecommerce conversation."""
    messages: List[ChatMessage]
    session_id: str
    user_preferences: Dict[str, float]
    extracted_intents: List[str]
    current_context: Dict[str, Any]
    recommendations: List[ProductRecommendation]
    graph_updates: List[Dict[str, Any]]
    follow_up_question: Optional[str]


class AgentState(MessagesState):
    """Extended state for the LangGraph agent."""
    session_id: str
    user_preferences: Dict[str, float]
    extracted_preferences: Dict[str, Any]
    product_context: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    graph_updates: List[Dict[str, Any]]
    follow_up_question: Optional[str]
    conversation_stage: str  # "greeting", "exploring", "refining", "concluding"