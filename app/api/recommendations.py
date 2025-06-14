"""Recommendations API endpoints."""

from fastapi import APIRouter, HTTPException, Request
from app.models.api_models import RecommendationRequest, ProductRecommendation, GraphStats
from typing import List

router = APIRouter()


@router.post("/", response_model=List[ProductRecommendation])
async def get_recommendations(request: RecommendationRequest, req: Request):
    """Get personalized recommendations for a user."""
    try:
        graph_service = req.app.state.graph_service
        chat_service = req.app.state.chat_service
        
        # Get recommendations
        recommendations = chat_service.recommendation_engine.get_product_recommendations(
            graph=graph_service.graph,
            user_id=f"user_{request.session_id}",
            context=request.context,
            limit=request.limit
        )
        
        # Convert to API format
        rec_list = []
        for product, score, reason in recommendations:
            rec_list.append(ProductRecommendation(
                product_id=product.id,
                name=product.name,
                category=product.category,
                style=product.style,
                price=product.price,
                rating=product.rating,
                colors=product.colors,
                brand=product.brand,
                score=round(score, 3),
                reason=reason
            ))
        
        return rec_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats", response_model=GraphStats)
async def get_graph_statistics(req: Request):
    """Get statistics about the recommendation graph."""
    try:
        graph_service = req.app.state.graph_service
        stats = graph_service.get_graph_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))