"""Products API endpoints."""

from fastapi import APIRouter, HTTPException, Request, Query
from app.models.api_models import ProductListResponse, ProductRecommendation, ProductFilter
from typing import Optional, List

router = APIRouter()


@router.get("/", response_model=ProductListResponse)
async def list_products(
    req: Request,
    search: Optional[str] = Query(None, description="Search products by name or description"),
    category: Optional[str] = Query(None, description="Filter by category"),
    style: Optional[str] = Query(None, description="Filter by style"),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    colors: Optional[List[str]] = Query(None, description="Filter by colors"),
    brands: Optional[List[str]] = Query(None, description="Filter by brands"),
    min_rating: Optional[float] = Query(None, description="Minimum rating")
):
    """Get all products with optional filters and search."""
    try:
        graph_service = req.app.state.graph_service
        chat_service = req.app.state.chat_service
        
        # Build filters
        filters = {}
        if category:
            filters["category"] = category
        if style:
            filters["style"] = style
        if min_price is not None:
            filters["min_price"] = min_price
        if max_price is not None:
            filters["max_price"] = max_price
        if colors:
            filters["colors"] = colors
        if brands:
            filters["brands"] = brands
        if min_rating is not None:
            filters["min_rating"] = min_rating
        if search:
            filters["search"] = search
        
        # Get products with search/filtering
        products = graph_service.get_all_products(filters)
        
        # If search term provided, score products by relevance
        if search:
            # Score products using recommendation engine's similarity logic
            scored_products = []
            for product in products:
                score = graph_service.calculate_search_relevance(product, search)
                scored_products.append((product, score))
            
            # Sort by relevance score (highest first)
            scored_products.sort(key=lambda x: x[1], reverse=True)
            products = [product for product, score in scored_products]
        
        # Convert to API format
        product_list = []
        for i, product in enumerate(products):
            # Calculate score based on search relevance or default
            if search:
                score = graph_service.calculate_search_relevance(product, search)
                reason = f"Matches '{search}'" if score > 0.5 else f"Related to '{search}'"
            else:
                score = 0.0
                reason = "Available product"
                
            product_list.append(ProductRecommendation(
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
        
        return ProductListResponse(
            products=product_list,
            total=len(product_list),
            filters_applied=filters
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{product_id}", response_model=ProductRecommendation)
async def get_product(product_id: str, req: Request):
    """Get a specific product by ID."""
    try:
        graph_service = req.app.state.graph_service
        
        product_node = graph_service.graph.nodes.get(product_id)
        if not product_node:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return ProductRecommendation(
            product_id=product_node.id,
            name=product_node.name,
            category=product_node.category,
            style=product_node.style,
            price=product_node.price,
            rating=product_node.rating,
            colors=product_node.colors,
            brand=product_node.brand,
            score=1.0,
            reason="Requested product"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{product_id}/similar", response_model=List[ProductRecommendation])
async def get_similar_products(
    product_id: str, 
    req: Request,
    limit: int = Query(5, description="Number of similar products to return")
):
    """Get products similar to a given product."""
    try:
        graph_service = req.app.state.graph_service
        chat_service = req.app.state.chat_service
        
        # Get similar products
        similar = chat_service.recommendation_engine.get_similar_products(
            graph=graph_service.graph,
            product_id=product_id,
            limit=limit
        )
        
        # Convert to API format
        recommendations = []
        for product, score in similar:
            recommendations.append(ProductRecommendation(
                product_id=product.id,
                name=product.name,
                category=product.category,
                style=product.style,
                price=product.price,
                rating=product.rating,
                colors=product.colors,
                brand=product.brand,
                score=round(score, 3),
                reason=f"Similar to {product_id}"
            ))
        
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))