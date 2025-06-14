"""Recommendation engine using GNN models."""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from app.gnn.models import HeteroGNN, ProductSimilarityModel, UserPreferenceModel
from app.models.graph_models import EcommerceGraph, ProductNode, UserNode, NodeType


class RecommendationEngine:
    """Engine for generating product recommendations using GNN."""
    
    def __init__(self):
        # Initialize models (no training needed for demo)
        self.gnn = HeteroGNN(
            metadata=(
                ['user', 'product', 'category', 'style', 'brand', 'color'],
                [
                    ('product', 'similar_to', 'product'),
                    ('product', 'belongs_to', 'category'),
                    ('product', 'has_style', 'style'),
                    ('user', 'prefers', 'product'),
                    ('user', 'viewed', 'product'),
                ]
            )
        )
        self.similarity_model = ProductSimilarityModel()
        self.preference_model = UserPreferenceModel()
        
        # Set to eval mode (no training)
        self.gnn.eval()
        self.similarity_model.eval()
        self.preference_model.eval()
        
    def compute_embeddings(self, graph: EcommerceGraph) -> Dict[str, torch.Tensor]:
        """Compute embeddings for all nodes in the graph."""
        with torch.no_grad():
            # Convert graph to HeteroData
            data = graph.to_hetero_data()
            
            # Get node features
            x_dict = {}
            for node_type in data.node_types:
                if hasattr(data[node_type], 'x'):
                    x_dict[node_type] = data[node_type].x
                else:
                    # Create dummy features if not present
                    num_nodes = data[node_type].num_nodes
                    x_dict[node_type] = torch.randn(num_nodes, 4)
            
            # Get edge indices
            edge_index_dict = {}
            for edge_type in data.edge_types:
                if hasattr(data[edge_type], 'edge_index'):
                    edge_index_dict[edge_type] = data[edge_type].edge_index
            
            # Forward pass through GNN
            embeddings = self.gnn(x_dict, edge_index_dict)
            
            return embeddings
    
    def get_product_recommendations(
        self, 
        graph: EcommerceGraph, 
        user_id: str, 
        context: Optional[str] = None,
        limit: int = 5
    ) -> List[Tuple[ProductNode, float, str]]:
        """
        Get product recommendations for a user.
        
        Returns:
            List of (product, score, reason) tuples
        """
        # Get user node
        user_node = graph.nodes.get(user_id)
        if not user_node or not isinstance(user_node, UserNode):
            return []
        
        # Compute embeddings
        embeddings = self.compute_embeddings(graph)
        
        # Get user and product embeddings
        user_embeddings = embeddings.get('user', torch.randn(1, 32))
        product_embeddings = embeddings.get('product', torch.randn(1, 32))
        
        # Map node IDs to embedding indices
        user_nodes = [n for n in graph.nodes.values() if n.type == NodeType.USER]
        product_nodes = [n for n in graph.nodes.values() if n.type == NodeType.PRODUCT]
        
        user_idx = next((i for i, n in enumerate(user_nodes) if n.id == user_id), 0)
        
        # Get user embedding
        user_emb = user_embeddings[user_idx] if user_idx < len(user_embeddings) else torch.randn(32)
        
        # Score all products
        recommendations = []
        
        for i, product_node in enumerate(product_nodes):
            if i >= len(product_embeddings):
                continue
                
            product_emb = product_embeddings[i]
            
            # Compute base similarity score
            similarity = torch.cosine_similarity(user_emb.unsqueeze(0), product_emb.unsqueeze(0)).item()
            
            # Boost score based on user preferences
            preference_boost = 0.0
            if hasattr(user_node, 'preferences'):
                # Check category preference
                if product_node.category in user_node.preferences:
                    preference_boost += user_node.preferences[product_node.category] * 0.3
                    
                # Check style preference
                if product_node.style in user_node.preferences:
                    preference_boost += user_node.preferences[product_node.style] * 0.3
                    
                # Check brand preference
                if product_node.brand in user_node.preferences:
                    preference_boost += user_node.preferences[product_node.brand] * 0.2
                    
                # Check color preferences
                for color in product_node.colors:
                    if color in user_node.preferences:
                        preference_boost += user_node.preferences[color] * 0.1
            
            # Combine scores
            final_score = (similarity + 1) / 2 * 0.5 + preference_boost * 0.5
            final_score = min(max(final_score, 0.0), 1.0)  # Clamp to [0, 1]
            
            # Generate reason
            reason = self._generate_recommendation_reason(
                product_node, user_node, final_score, similarity, preference_boost
            )
            
            recommendations.append((product_node, final_score, reason))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:limit]
    
    def get_similar_products(
        self, 
        graph: EcommerceGraph, 
        product_id: str, 
        limit: int = 5
    ) -> List[Tuple[ProductNode, float]]:
        """Get products similar to a given product."""
        # First check if there are explicit similar_to edges
        similar = graph.get_similar_products(product_id, limit)
        if similar:
            return similar
        
        # Otherwise compute similarity using embeddings
        product_node = graph.nodes.get(product_id)
        if not product_node or not isinstance(product_node, ProductNode):
            return []
        
        # Compute embeddings
        embeddings = self.compute_embeddings(graph)
        product_embeddings = embeddings.get('product', torch.randn(1, 32))
        
        # Get all product nodes
        product_nodes = [n for n in graph.nodes.values() if n.type == NodeType.PRODUCT]
        
        # Find source product index
        source_idx = next((i for i, n in enumerate(product_nodes) if n.id == product_id), None)
        if source_idx is None or source_idx >= len(product_embeddings):
            return []
        
        source_emb = product_embeddings[source_idx]
        
        # Compute similarities
        similarities = []
        for i, candidate_node in enumerate(product_nodes):
            if i == source_idx or i >= len(product_embeddings):
                continue
                
            candidate_emb = product_embeddings[i]
            sim = torch.cosine_similarity(source_emb.unsqueeze(0), candidate_emb.unsqueeze(0)).item()
            
            # Boost similarity for same category/style
            if candidate_node.category == product_node.category:
                sim += 0.1
            if candidate_node.style == product_node.style:
                sim += 0.1
                
            similarities.append((candidate_node, min(sim, 1.0)))
        
        # Sort and return top similar products
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:limit]
    
    def _generate_recommendation_reason(
        self, 
        product: ProductNode, 
        user: UserNode, 
        score: float,
        similarity: float,
        preference_boost: float
    ) -> str:
        """Generate human-readable reason for recommendation."""
        reasons = []
        
        # High score reason
        if score > 0.8:
            reasons.append("Highly recommended based on your preferences")
        elif score > 0.6:
            reasons.append("Good match for your style")
        
        # Preference-based reasons
        if hasattr(user, 'preferences'):
            if product.style in user.preferences and user.preferences[product.style] > 0.5:
                reasons.append(f"matches your {product.style} style preference")
                
            if product.category in user.preferences and user.preferences[product.category] > 0.5:
                reasons.append(f"from your preferred {product.category} category")
                
            # Check for color preferences
            matching_colors = [c for c in product.colors if c in user.preferences and user.preferences[c] > 0.3]
            if matching_colors:
                reasons.append(f"available in your preferred color{'s' if len(matching_colors) > 1 else ''}: {', '.join(matching_colors)}")
        
        # Price/rating reasons
        if product.rating >= 4.5:
            reasons.append("highly rated by customers")
        
        # Similarity reason
        if similarity > 0.7:
            reasons.append("similar to items you've shown interest in")
        
        # Default reason
        if not reasons:
            reasons.append("selected based on your shopping patterns")
        
        # Combine reasons
        if len(reasons) == 1:
            return reasons[0].capitalize()
        else:
            return reasons[0].capitalize() + " and " + ", ".join(reasons[1:])
    
    def update_from_interaction(
        self, 
        graph: EcommerceGraph, 
        user_id: str, 
        product_id: str, 
        interaction_type: str
    ):
        """Update embeddings based on user interaction (view, purchase, etc)."""
        # In a real system, this would trigger model updates
        # For the demo, we just ensure the graph structure is updated
        pass