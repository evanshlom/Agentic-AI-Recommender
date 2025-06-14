"""Tests for the LangGraph agent."""

import pytest
from unittest.mock import Mock, AsyncMock
from app.agents.ecommerce_agent import EcommerceAgent
from app.agents.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = Mock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def mock_graph_service():
    """Mock graph service for testing."""
    service = Mock()
    service.get_user_node = Mock()
    service.add_interaction = Mock()
    return service


@pytest.fixture
def mock_recommendation_engine():
    """Mock recommendation engine for testing."""
    engine = Mock()
    engine.get_product_recommendations = Mock(return_value=[])
    return engine


@pytest.fixture
def agent(mock_llm, mock_graph_service, mock_recommendation_engine):
    """Create agent with mocked dependencies."""
    return EcommerceAgent(mock_llm, mock_graph_service, mock_recommendation_engine)


@pytest.mark.asyncio
async def test_analyze_user_input_greeting(agent):
    """Test analyzing first message."""
    state = {
        "messages": [HumanMessage(content="Hi, I'm looking for clothes")],
        "session_id": "test-123"
    }
    
    result = await agent.analyze_user_input(state)
    
    assert result["conversation_stage"] == "greeting"
    assert result["extracted_preferences"]["intent"] == "initial_greeting"


@pytest.mark.asyncio
async def test_analyze_user_input_preferences(agent, mock_llm):
    """Test extracting preferences from user input."""
    mock_llm.ainvoke.return_value = Mock(
        content='{"intent": "shopping", "preferences": {"categories": ["shirts"], "styles": ["casual"]}, "confidence": 0.8}'
    )
    
    state = {
        "messages": [
            HumanMessage(content="Hi"),
            AIMessage(content="Welcome!"),
            HumanMessage(content="I need casual shirts")
        ],
        "session_id": "test-123",
        "conversation_stage": "exploring"
    }
    
    result = await agent.analyze_user_input(state)
    
    assert result["extracted_preferences"]["intent"] == "shopping"
    assert "shirts" in result["extracted_preferences"]["preferences"]["categories"]
    assert "casual" in result["extracted_preferences"]["preferences"]["styles"]


@pytest.mark.asyncio
async def test_update_user_graph(agent, mock_graph_service):
    """Test updating user preferences in graph."""
    state = {
        "extracted_preferences": {
            "preferences": {
                "categories": ["shirts"],
                "styles": ["business-casual"],
                "colors": ["blue", "white"]
            }
        },
        "user_preferences": {},
        "session_id": "test-123"
    }
    
    result = await agent.update_user_graph(state)
    
    assert "shirts" in result["user_preferences"]
    assert result["user_preferences"]["shirts"] == 0.3
    assert "business-casual" in result["user_preferences"]
    assert result["user_preferences"]["business-casual"] == 0.4
    assert len(result["graph_updates"]) == 4  # 1 category + 1 style + 2 colors


@pytest.mark.asyncio
async def test_generate_response_greeting(agent):
    """Test generating greeting response."""
    state = {
        "messages": [HumanMessage(content="Hi")],
        "conversation_stage": "greeting",
        "session_id": "test-123"
    }
    
    result = await agent.generate_response(state)
    
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][-1], AIMessage)
    assert "Welcome" in result["messages"][-1].content or "Hi" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_process_message(agent, mock_llm, mock_recommendation_engine):
    """Test processing a complete message."""
    mock_llm.ainvoke.return_value = Mock(
        content='{"intent": "shopping", "preferences": {"categories": ["shirts"]}, "confidence": 0.8}'
    )
    
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
    
    mock_recommendation_engine.get_product_recommendations.return_value = [
        (mock_product, 0.9, "Great match for your style")
    ]
    
    message, recommendations, follow_up = await agent.process_message(
        "I need shirts",
        "test-123"
    )
    
    assert isinstance(message, str)
    assert len(recommendations) > 0
    assert recommendations[0].product_id == "prod-001"