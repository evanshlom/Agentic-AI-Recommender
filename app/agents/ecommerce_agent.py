"""LangGraph agent for ecommerce chatbot."""

import json
import random
from typing import Dict, Any, List, Tuple, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from app.agents.state import AgentState
from app.agents.prompts import (
    ANALYZE_INPUT_PROMPT, 
    GENERATE_RESPONSE_PROMPT,
    GENERATE_FOLLOW_UP_PROMPT,
    GREETING_VARIATIONS
)
from app.models.graph_models import EdgeType
from app.gnn.engine import RecommendationEngine
from app.models.api_models import ProductRecommendation


class EcommerceAgent:
    """LangGraph agent for handling ecommerce conversations."""
    
    def __init__(self, llm: ChatOpenAI, graph_service, recommendation_engine: RecommendationEngine):
        self.llm = llm
        self.graph_service = graph_service
        self.recommendation_engine = recommendation_engine
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_input", self.analyze_user_input)
        workflow.add_node("update_graph", self.update_user_graph)
        workflow.add_node("generate_recommendations", self.generate_recommendations)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("update_state", self.update_conversation_state)
        
        # Add edges
        workflow.set_entry_point("analyze_input")
        workflow.add_edge("analyze_input", "update_graph")
        workflow.add_edge("update_graph", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "generate_response")
        workflow.add_edge("generate_response", "update_state")
        workflow.add_edge("update_state", END)
        
        return workflow.compile()
    
    async def analyze_user_input(self, state: AgentState) -> AgentState:
        """Analyze user input to extract preferences and intent."""
        messages = state.get("messages", [])
        if not messages:
            return state
            
        last_message = messages[-1]
        
        # Check if this is the first message
        if len(messages) == 1:
            state["conversation_stage"] = "greeting"
            state["extracted_preferences"] = {
                "intent": "initial_greeting",
                "preferences": {},
                "needs_clarification": []
            }
            return state
        
        # Extract preferences using LLM
        prompt = ANALYZE_INPUT_PROMPT.format(
            message=last_message.content,
            context=json.dumps(state.get("product_context", {}))
        )
        
        response = await self.llm.ainvoke(prompt)
        
        try:
            extracted = json.loads(response.content)
        except:
            # Fallback extraction
            extracted = {
                "intent": "browsing",
                "preferences": self._extract_basic_preferences(last_message.content),
                "confidence": 0.5,
                "needs_clarification": []
            }
        
        state["extracted_preferences"] = extracted
        
        # Update conversation stage
        if state["conversation_stage"] == "greeting":
            state["conversation_stage"] = "exploring"
        elif len(state.get("user_preferences", {})) > 5:
            state["conversation_stage"] = "refining"
            
        return state
    
    async def update_user_graph(self, state: AgentState) -> AgentState:
        """Update the user's preference graph based on extracted information."""
        extracted = state.get("extracted_preferences", {})
        preferences = extracted.get("preferences", {})
        
        # Initialize graph updates list
        graph_updates = []
        user_preferences = state.get("user_preferences", {})
        
        # Update preferences with weights
        for category in preferences.get("categories", []):
            weight = user_preferences.get(category, 0.0) + 0.3
            user_preferences[category] = min(weight, 1.0)
            graph_updates.append({
                "type": "preference",
                "key": category,
                "value": user_preferences[category]
            })
            
        for style in preferences.get("styles", []):
            weight = user_preferences.get(style, 0.0) + 0.4
            user_preferences[style] = min(weight, 1.0)
            graph_updates.append({
                "type": "preference", 
                "key": style,
                "value": user_preferences[style]
            })
            
        for color in preferences.get("colors", []):
            weight = user_preferences.get(color, 0.0) + 0.2
            user_preferences[color] = min(weight, 1.0)
            graph_updates.append({
                "type": "preference",
                "key": color,
                "value": user_preferences[color]
            })
            
        # Update state
        state["user_preferences"] = user_preferences
        state["graph_updates"] = graph_updates
        
        # Actually update the graph
        if self.graph_service:
            user_node = self.graph_service.get_user_node(state["session_id"])
            if user_node:
                for pref, weight in user_preferences.items():
                    user_node.update_preference(pref, weight)
                    
        return state
    
    async def generate_recommendations(self, state: AgentState) -> AgentState:
        """Generate product recommendations using the GNN engine."""
        # Get recommendations from the recommendation engine
        recommendations = self.recommendation_engine.get_product_recommendations(
            graph=self.graph_service.graph,
            user_id=f"user_{state['session_id']}",
            context=json.dumps(state.get("extracted_preferences", {})),
            limit=5
        )
        
        # Convert to API format
        rec_list = []
        for product, score, reason in recommendations:
            rec_list.append({
                "product_id": product.id,
                "name": product.name,
                "category": product.category,
                "style": product.style,
                "price": product.price,
                "rating": product.rating,
                "colors": product.colors,
                "brand": product.brand,
                "score": round(score, 3),
                "reason": reason
            })
            
        state["recommendations"] = rec_list
        
        # Track product views
        if self.graph_service:
            for rec in rec_list[:3]:  # Track top 3 as viewed
                self.graph_service.add_interaction(
                    user_id=f"user_{state['session_id']}",
                    product_id=rec["product_id"],
                    interaction_type="viewed"
                )
                
        return state
    
    async def generate_response(self, state: AgentState) -> AgentState:
        """Generate the conversational response with recommendations."""
        stage = state.get("conversation_stage", "greeting")
        
        # Handle greeting specially
        if stage == "greeting":
            response = random.choice(GREETING_VARIATIONS)
            state["messages"].append(AIMessage(content=response))
            state["follow_up_question"] = None
            return state
        
        # Generate response based on recommendations
        recommendations = state.get("recommendations", [])
        preferences = state.get("user_preferences", {})
        
        # Create response prompt
        prompt = GENERATE_RESPONSE_PROMPT.format(
            stage=stage,
            preferences=json.dumps(preferences),
            recommendations=json.dumps(recommendations[:3])  # Top 3 for response
        )
        
        response = await self.llm.ainvoke(prompt)
        response_text = response.content
        
        # Generate follow-up question
        follow_up_prompt = GENERATE_FOLLOW_UP_PROMPT.format(
            preferences=json.dumps(preferences),
            stage=stage,
            message=state["messages"][-1].content if state["messages"] else ""
        )
        
        follow_up_response = await self.llm.ainvoke(follow_up_prompt)
        follow_up = follow_up_response.content.strip()
        
        # Combine response with follow-up
        full_response = f"{response_text}\n\n{follow_up}"
        
        state["messages"].append(AIMessage(content=full_response))
        state["follow_up_question"] = follow_up
        
        return state
    
    async def update_conversation_state(self, state: AgentState) -> AgentState:
        """Update the conversation state for the next turn."""
        # Update product context based on recommendations shown
        if state.get("recommendations"):
            state["product_context"] = {
                "last_shown": [r["name"] for r in state["recommendations"][:3]],
                "categories": list(set(r["category"] for r in state["recommendations"])),
                "price_range": {
                    "min": min(r["price"] for r in state["recommendations"]),
                    "max": max(r["price"] for r in state["recommendations"])
                }
            }
            
        return state
    
    def _extract_basic_preferences(self, text: str) -> Dict[str, List[str]]:
        """Basic preference extraction without LLM."""
        text_lower = text.lower()
        
        # Simple keyword matching
        categories = []
        if any(word in text_lower for word in ["shirt", "shirts", "top", "blouse"]):
            categories.append("shirts")
        if any(word in text_lower for word in ["pant", "pants", "trouser", "jeans"]):
            categories.append("pants")
        if any(word in text_lower for word in ["dress", "dresses", "gown"]):
            categories.append("dresses")
        if any(word in text_lower for word in ["shoe", "shoes", "sneaker", "boot"]):
            categories.append("shoes")
        if any(word in text_lower for word in ["jacket", "coat", "blazer"]):
            categories.append("jackets")
            
        styles = []
        if any(word in text_lower for word in ["casual", "relaxed", "comfortable"]):
            styles.append("casual")
        if any(word in text_lower for word in ["business", "formal", "professional"]):
            styles.append("business")
        if any(word in text_lower for word in ["athletic", "sport", "gym", "workout"]):
            styles.append("athletic")
            
        colors = []
        for color in ["black", "white", "blue", "red", "green", "gray", "navy", "pink"]:
            if color in text_lower:
                colors.append(color)
                
        return {
            "categories": categories,
            "styles": styles,
            "colors": colors,
            "brands": [],
            "price_range": {"min": None, "max": None}
        }
    
    async def process_message(self, message: str, session_id: str) -> Tuple[str, List[ProductRecommendation], Optional[str]]:
        """Process a message and return response, recommendations, and follow-up."""
        # Initialize state
        state = {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "user_preferences": {},
            "extracted_preferences": {},
            "product_context": {},
            "recommendations": [],
            "graph_updates": [],
            "follow_up_question": None,
            "conversation_stage": "greeting"
        }
        
        # Run the workflow
        result = await self.workflow.ainvoke(state)
        
        # Extract response
        response_message = result["messages"][-1].content if result["messages"] else ""
        
        # Convert recommendations
        recommendations = []
        for rec in result.get("recommendations", []):
            recommendations.append(ProductRecommendation(**rec))
            
        return response_message, recommendations, result.get("follow_up_question")