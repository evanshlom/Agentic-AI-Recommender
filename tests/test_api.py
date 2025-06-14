"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from app.main import app
from app.models.api_models import ChatResponse, ProductRecommendation


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock services for testing."""
    with patch('app.main.graph_service') as mock_graph, \
         patch('app.main.chat_service') as mock_chat:
        
        # Setup graph service
        mock_graph.get_graph_stats.return_value = Mock(
            total_nodes=100,
            total_edges=200,
            node_counts={"product": 50, "user": 10},
            edge_counts={"similar_to": 100},
            active_sessions=5
        )
        
        # Setup chat service
        mock_chat.process_chat = AsyncMock()
        
        yield mock_graph, mock_chat


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_chat_endpoint(client, mock_services):
    """Test chat endpoint."""
    mock_graph, mock_chat = mock_services
    
    # Mock chat response
    mock_chat.process_chat.return_value = ChatResponse(
        message="I found some great shirts for you!",
        recommendations=[
            ProductRecommendation(
                product_id="prod-001",
                name="Test Shirt",
                category="shirts",
                style="casual",
                price=50.0,
                rating=4.5,
                colors=["blue"],
                brand="TestBrand",
                score=0.9,
                reason="Perfect match"
            )
        ],
        session_id="test-123",
        follow_up_question="What's your preferred size?"
    )
    
    response = client.post("/api/chat/", json={
        "message": "I need shirts",
        "session_id": "test-123"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) == 1
    assert data["session_id"] == "test-123"


def test_get_products(client, mock_services):
    """Test products listing endpoint."""
    mock_graph, _ = mock_services
    
    # Mock products
    mock_product = Mock(
        id="prod-001",
        name="Test Shirt",
        category="shirts",
        style="casual",
        price=50.0,
        rating=4.5,
        colors=["blue"],
        brand="TestBrand"
    )
    
    mock_graph.get_all_products.return_value = [mock_product]
    
    response = client.get("/api/products/")
    
    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    assert "total" in data
    assert data["total"] == 1


def test_get_products_with_filters(client, mock_services):
    """Test products listing with filters."""
    mock_graph, _ = mock_services
    mock_graph.get_all_products.return_value = []
    
    response = client.get("/api/products/?category=shirts&max_price=100")
    
    assert response.status_code == 200
    mock_graph.get_all_products.assert_called_once()
    call_args = mock_graph.get_all_products.call_args[0][0]
    assert call_args["category"] == "shirts"
    assert call_args["max_price"] == 100


def test_get_graph_stats(client, mock_services):
    """Test graph statistics endpoint."""
    response = client.get("/api/recommendations/graph/stats")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_nodes"] == 100
    assert data["total_edges"] == 200
    assert "node_counts" in data
    assert "edge_counts" in data


def test_error_handling(client, mock_services):
    """Test error handling in endpoints."""
    mock_graph, mock_chat = mock_services
    mock_chat.process_chat.side_effect = Exception("Test error")
    
    response = client.post("/api/chat/", json={
        "message": "test"
    })
    
    assert response.status_code == 500
    assert "detail" in response.json()