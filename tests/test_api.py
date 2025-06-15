"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
from app.main import app
from app.models.api_models import ChatResponse, ProductRecommendation


@pytest.fixture
def client():
    """Create test client with dependency overrides."""
    # Create mock services
    mock_graph = Mock()
    mock_chat = Mock()

    # Configure mock behaviors
    mock_graph.get_graph_stats.return_value = {
        "total_nodes": 100,
        "total_edges": 200,
        "node_counts": {"product": 50, "user": 10},
        "edge_counts": {"similar_to": 100},
        "active_sessions": 5
    }
    mock_graph.get_all_products.return_value = []
    mock_chat.process_chat = AsyncMock()

    # Override app state with mocks
    app.state.graph_service = mock_graph
    app.state.chat_service = mock_chat

    client = TestClient(app)

    # Attach mocks for test access
    client.mock_graph = mock_graph
    client.mock_chat = mock_chat

    yield client

    # Cleanup (optional)
    if hasattr(app.state, 'graph_service'):
        delattr(app.state, 'graph_service')
    if hasattr(app.state, 'chat_service'):
        delattr(app.state, 'chat_service')


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
async def test_chat_endpoint(client):
    """Test chat endpoint."""
    # Configure mock response
    client.mock_chat.process_chat.return_value = ChatResponse(
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


def test_get_products(client):
    """Test products listing endpoint."""
    # Mock products with actual attribute values
    from unittest.mock import Mock

    mock_product = Mock()
    # Set actual values for attributes (not Mock objects)
    mock_product.id = "prod-001"
    mock_product.name = "Test Shirt"
    mock_product.category = "shirts"
    mock_product.style = "casual"
    mock_product.price = 50.0
    mock_product.rating = 4.5
    mock_product.colors = ["blue"]
    mock_product.brand = "TestBrand"

    client.mock_graph.get_all_products.return_value = [mock_product]

    response = client.get("/api/products/")

    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    assert "total" in data
    assert data["total"] == 1


def test_get_products_with_filters(client):
    """Test products listing with filters."""
    client.mock_graph.get_all_products.return_value = []

    response = client.get("/api/products/?category=shirts&max_price=100")

    assert response.status_code == 200
    data = response.json()
    assert "products" in data
    assert "total" in data
    assert data["total"] == 0


def test_get_graph_stats(client):
    """Test graph statistics endpoint."""
    response = client.get("/api/recommendations/graph/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["total_nodes"] == 100
    assert data["total_edges"] == 200
    assert "node_counts" in data
    assert "edge_counts" in data


def test_error_handling(client):
    """Test error handling in endpoints."""
    # Configure mock to throw exception
    client.mock_chat.process_chat.side_effect = Exception("Test error")

    response = client.post("/api/chat/", json={
        "message": "test",
        "session_id": "test-123"
    })

    # Reset for other tests
    client.mock_chat.process_chat.side_effect = None
    client.mock_chat.process_chat.return_value = ChatResponse(
        message="test", recommendations=[], session_id="test-123"
    )

    # Should handle error gracefully
    # Either server error or validation error
    assert response.status_code in [500, 422]
