"""Tests for GNN models and recommendation engine."""

import pytest
import torch
from app.gnn.models import HeteroGNN, ProductSimilarityModel
from app.gnn.engine import RecommendationEngine
from app.models.graph_models import EcommerceGraph, ProductNode, UserNode, GraphEdge, NodeType, EdgeType


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    graph = EcommerceGraph()
    
    # Add user
    user = UserNode(
        id="user_test",
        session_id="test",
        preferences={"casual": 0.8, "blue": 0.6}
    )
    graph.add_node(user)
    
    # Add products
    products = [
        ProductNode(
            id="prod_1",
            name="Casual Shirt",
            category="shirts",
            style="casual",
            price=50.0,
            colors=["blue", "white"],
            sizes=["M", "L"],
            brand="Brand1",
            rating=4.5
        ),
        ProductNode(
            id="prod_2",
            name="Business Shirt",
            category="shirts",
            style="business",
            price=80.0,
            colors=["white", "gray"],
            sizes=["M", "L"],
            brand="Brand2",
            rating=4.7
        )
    ]
    
    for product in products:
        graph.add_node(product)
    
    # Add edges
    graph.add_edge(GraphEdge(
        source="prod_1",
        target="prod_2",
        edge_type=EdgeType.SIMILAR_TO,
        weight=0.7
    ))
    
    graph.add_edge(GraphEdge(
        source="user_test",
        target="prod_1",
        edge_type=EdgeType.VIEWED,
        weight=1.0
    ))
    
    return graph


def test_hetero_gnn_initialization():
    """Test HeteroGNN model initialization."""
    metadata = (
        ['user', 'product', 'category'],
        [('product', 'similar_to', 'product'), ('user', 'prefers', 'product')]
    )
    
    model = HeteroGNN(metadata)
    
    assert 'user' in model.node_projections
    assert 'product' in model.node_projections
    assert 'category' in model.node_projections


def test_hetero_gnn_forward():
    """Test HeteroGNN forward pass."""
    metadata = (
        ['user', 'product'],
        [('product', 'similar_to', 'product')]
    )
    
    model = HeteroGNN(metadata, hidden_dim=32, out_dim=16)
    model.eval()
    
    # Create dummy data
    x_dict = {
        'user': torch.randn(2, 4),
        'product': torch.randn(5, 4)
    }
    
    edge_index_dict = {
        ('product', 'similar_to', 'product'): torch.tensor([[0, 1], [1, 2]]).t()
    }
    
    with torch.no_grad():
        out = model(x_dict, edge_index_dict)
    
    assert 'user' in out
    assert 'product' in out
    assert out['product'].shape == (5, 16)


def test_product_similarity_model():
    """Test ProductSimilarityModel."""
    model = ProductSimilarityModel(embedding_dim=32)
    model.eval()
    
    product_emb = torch.randn(32)
    candidate_embs = torch.randn(10, 32)
    
    with torch.no_grad():
        similarities = model(product_emb, candidate_embs)
    
    assert similarities.shape == (10,)
    assert torch.all(similarities >= -1) and torch.all(similarities <= 1)


def test_recommendation_engine_initialization():
    """Test RecommendationEngine initialization."""
    engine = RecommendationEngine()
    
    assert engine.gnn is not None
    assert engine.similarity_model is not None
    assert engine.preference_model is not None


def test_recommendation_engine_compute_embeddings(sample_graph):
    """Test computing embeddings for a graph."""
    engine = RecommendationEngine()
    
    embeddings = engine.compute_embeddings(sample_graph)
    
    assert 'user' in embeddings
    assert 'product' in embeddings
    assert embeddings['product'].shape[0] >= 2  # At least 2 products


def test_recommendation_engine_get_recommendations(sample_graph):
    """Test getting product recommendations."""
    engine = RecommendationEngine()
    
    recommendations = engine.get_product_recommendations(
        graph=sample_graph,
        user_id="user_test",
        limit=2
    )
    
    assert len(recommendations) <= 2
    for product, score, reason in recommendations:
        assert isinstance(product, ProductNode)
        assert 0 <= score <= 1
        assert isinstance(reason, str)


def test_recommendation_engine_similar_products(sample_graph):
    """Test finding similar products."""
    engine = RecommendationEngine()
    
    similar = engine.get_similar_products(
        graph=sample_graph,
        product_id="prod_1",
        limit=1
    )
    
    assert len(similar) <= 1
    if similar:
        product, score = similar[0]
        assert isinstance(product, ProductNode)
        assert 0 <= score <= 1


def test_recommendation_reason_generation():
    """Test recommendation reason generation."""
    engine = RecommendationEngine()
    
    product = ProductNode(
        id="test",
        name="Test Product",
        category="shirts",
        style="casual",
        price=50.0,
        colors=["blue"],
        sizes=["M"],
        brand="TestBrand",
        rating=4.8
    )
    
    user = UserNode(
        id="user_test",
        session_id="test",
        preferences={"casual": 0.9, "blue": 0.8}
    )
    
    reason = engine._generate_recommendation_reason(
        product, user, score=0.9, similarity=0.8, preference_boost=0.7
    )
    
    assert isinstance(reason, str)
    assert len(reason) > 0