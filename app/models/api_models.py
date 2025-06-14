"""Pydantic models for API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "I'm looking for business casual shirts",
                "session_id": "user-123"
            }
        }
    )


class ProductRecommendation(BaseModel):
    """Product recommendation with explanation."""
    product_id: str
    name: str
    category: str
    style: str
    price: float
    rating: float
    colors: List[str]
    brand: str
    score: float = Field(..., description="Recommendation score (0-1)")
    reason: str = Field(..., description="Why this product was recommended")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_id": "prod-001",
                "name": "Oxford Business Shirt",
                "category": "shirts",
                "style": "business-casual",
                "price": 79.99,
                "rating": 4.5,
                "colors": ["white", "light-blue"],
                "brand": "ProWear",
                "score": 0.95,
                "reason": "Perfect match for business casual style with excellent ratings"
            }
        }
    )


class ChatResponse(BaseModel):
    """Chat response model."""
    message: str = Field(..., description="Assistant's response")
    recommendations: List[ProductRecommendation] = Field(default_factory=list)
    session_id: str = Field(..., description="Session ID for tracking")
    follow_up_question: Optional[str] = Field(
        None, description="Optional follow-up to keep conversation flowing")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "I found some great business casual shirts that match your style!",
                "recommendations": [],
                "session_id": "user-123",
                "follow_up_question": "Do you have a preferred color or price range in mind?"
            }
        }
    )


class ProductFilter(BaseModel):
    """Filters for product search."""
    category: Optional[str] = None
    style: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    colors: Optional[List[str]] = None
    brands: Optional[List[str]] = None
    min_rating: Optional[float] = None


class ProductListResponse(BaseModel):
    """Response for product listing."""
    products: List[ProductRecommendation]
    total: int
    filters_applied: Dict[str, Any]


class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    session_id: str = Field(..., description="User session ID")
    limit: int = Field(5, description="Number of recommendations")
    context: Optional[str] = Field(
        None, description="Additional context for recommendations")


class GraphStats(BaseModel):
    """Graph statistics."""
    total_nodes: int
    total_edges: int
    node_counts: Dict[str, int]
    edge_counts: Dict[str, int]
    active_sessions: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_nodes": 150,
                "total_edges": 450,
                "node_counts": {
                    "product": 100,
                    "user": 5,
                    "category": 10,
                    "style": 15,
                    "brand": 10,
                    "color": 10
                },
                "edge_counts": {
                    "similar_to": 200,
                    "belongs_to": 100,
                    "has_style": 100,
                    "prefers": 50
                },
                "active_sessions": 3
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"
    services: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    status_code: int
