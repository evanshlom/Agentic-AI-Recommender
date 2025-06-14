"""Embedding generation and management for GNN."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from app.models.graph_models import BaseNode, ProductNode, UserNode


class EmbeddingGenerator:
    """Generate embeddings for different node types."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.encoders = self._initialize_encoders()
        
    def _initialize_encoders(self) -> Dict[str, nn.Module]:
        """Initialize encoders for different attribute types."""
        return {
            'text': nn.Sequential(
                nn.Linear(768, 256),  # Assuming BERT-like embeddings
                nn.ReLU(),
                nn.Linear(256, self.embedding_dim)
            ),
            'categorical': nn.Embedding(1000, self.embedding_dim),  # Max 1000 categories
            'numerical': nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, self.embedding_dim)
            )
        }
    
    def generate_product_embedding(self, product: ProductNode) -> torch.Tensor:
        """Generate embedding for a product node."""
        # Simple feature-based embedding (no pre-training needed)
        features = []
        
        # Encode price (normalized)
        price_norm = min(product.price / 200.0, 1.0)
        features.append(price_norm)
        
        # Encode rating
        rating_norm = product.rating / 5.0
        features.append(rating_norm)
        
        # Encode category (simple hash)
        category_hash = hash(product.category) % 100 / 100.0
        features.append(category_hash)
        
        # Encode style
        style_hash = hash(product.style) % 100 / 100.0
        features.append(style_hash)
        
        # Encode brand
        brand_hash = hash(product.brand) % 100 / 100.0
        features.append(brand_hash)
        
        # Encode number of colors
        colors_norm = len(product.colors) / 10.0
        features.append(colors_norm)
        
        # Encode number of sizes
        sizes_norm = len(product.sizes) / 10.0
        features.append(sizes_norm)
        
        # Pad to embedding dimension
        while len(features) < self.embedding_dim:
            features.append(0.0)
        
        return torch.tensor(features[:self.embedding_dim], dtype=torch.float32)
    
    def generate_user_embedding(self, user: UserNode) -> torch.Tensor:
        """Generate embedding for a user node based on preferences."""
        embedding = torch.zeros(self.embedding_dim)
        
        # Encode preferences into embedding
        for i, (pref, weight) in enumerate(user.preferences.items()):
            if i >= self.embedding_dim // 2:
                break
            # Use preference name hash for position
            pos = hash(pref) % (self.embedding_dim // 2)
            embedding[pos] = weight
            
        # Add interaction history features
        if user.interaction_history:
            interaction_count = len(user.interaction_history)
            embedding[-1] = min(interaction_count / 10.0, 1.0)
            
        return embedding
    
    def generate_category_embedding(self, category: str) -> torch.Tensor:
        """Generate embedding for a category."""
        # Simple deterministic embedding based on category name
        embedding = torch.zeros(self.embedding_dim)
        
        # Use character-based features
        for i, char in enumerate(category[:self.embedding_dim // 4]):
            embedding[i * 4] = ord(char) / 128.0
            
        # Add category-specific patterns
        if 'casual' in category.lower():
            embedding[self.embedding_dim // 2] = 0.8
        elif 'business' in category.lower():
            embedding[self.embedding_dim // 2 + 1] = 0.8
        elif 'athletic' in category.lower():
            embedding[self.embedding_dim // 2 + 2] = 0.8
            
        return embedding
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings."""
        return torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    
    def batch_embed_nodes(self, nodes: List[BaseNode]) -> torch.Tensor:
        """Generate embeddings for a batch of nodes."""
        embeddings = []
        
        for node in nodes:
            if isinstance(node, ProductNode):
                emb = self.generate_product_embedding(node)
            elif isinstance(node, UserNode):
                emb = self.generate_user_embedding(node)
            else:
                # Default embedding for other node types
                emb = torch.randn(self.embedding_dim) * 0.1
                
            embeddings.append(emb)
            
        return torch.stack(embeddings)