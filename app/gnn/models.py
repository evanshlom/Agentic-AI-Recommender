"""Graph Neural Network models for product recommendations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData
from typing import Dict, Tuple, List


class HeteroGNN(nn.Module):
    """Heterogeneous Graph Neural Network for ecommerce recommendations."""

    def __init__(self, metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 hidden_dim: int = 64, out_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Linear projections for each node type to hidden dimension
        self.node_projections = nn.ModuleDict()
        node_types = metadata[0] if metadata else []

        # Default input dimensions for each node type
        input_dims = {
            'user': 4,
            'product': 4,
            'category': 4,
            'style': 4,
            'brand': 4,
            'color': 4
        }

        for node_type in node_types:
            self.node_projections[node_type] = Linear(
                input_dims.get(node_type, 4),
                hidden_dim
            )

        # First heterogeneous convolution layer
        self.conv1 = HeteroConv({
            ('product', 'similar_to', 'product'): GATConv(
                hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=True
            ),
            ('product', 'belongs_to', 'category'): GATConv(
                hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False
            ),
            ('product', 'has_style', 'style'): GATConv(
                hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False
            ),
            ('user', 'prefers', 'product'): GATConv(
                hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False
            ),
            ('user', 'viewed', 'product'): GATConv(
                hidden_dim, hidden_dim, heads=num_heads, concat=False, add_self_loops=False
            ),
        }, aggr='mean')

        # Second heterogeneous convolution layer
        self.conv2 = HeteroConv({
            ('product', 'similar_to', 'product'): GATConv(
                hidden_dim, out_dim, heads=1, concat=False, add_self_loops=True
            ),
            ('product', 'belongs_to', 'category'): GATConv(
                hidden_dim, out_dim, heads=1, concat=False, add_self_loops=False
            ),
            ('user', 'prefers', 'product'): GATConv(
                hidden_dim, out_dim, heads=1, concat=False, add_self_loops=False
            ),
        }, aggr='mean')

        # Output projections for each node type
        self.node_outputs = nn.ModuleDict()
        for node_type in node_types:
            self.node_outputs[node_type] = Linear(out_dim, out_dim)

    def forward(self, x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the heterogeneous GNN."""

        # Project all node types to hidden dimension
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.node_projections:
                h_dict[node_type] = F.relu(self.node_projections[node_type](x))
            else:
                h_dict[node_type] = x

        # First convolution layer
        h_dict = self.conv1(h_dict, edge_index_dict)
        h_dict = {key: F.relu(h) for key, h in h_dict.items()}

        # Second convolution layer
        h_dict = self.conv2(h_dict, edge_index_dict)

        # Output projections
        out_dict = {}
        for node_type, h in h_dict.items():
            if node_type in self.node_outputs:
                out_dict[node_type] = self.node_outputs[node_type](h)
            else:
                out_dict[node_type] = h

        return out_dict


class ProductSimilarityModel(nn.Module):
    """Model for computing product similarities based on embeddings."""

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Attention mechanism for computing importance weights
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, product_embedding: torch.Tensor,
                candidate_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarities between a product and multiple candidates.

        Args:
            product_embedding: Shape (embedding_dim,)
            candidate_embeddings: Shape (num_candidates, embedding_dim)

        Returns:
            similarities: Shape (num_candidates,)
        """
        # Expand product embedding to match candidates
        product_exp = product_embedding.unsqueeze(
            0).expand_as(candidate_embeddings)

        # Concatenate for attention
        combined = torch.cat([product_exp, candidate_embeddings], dim=1)
        attention_weights = self.attention(combined).squeeze()

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(product_exp, candidate_embeddings, dim=1)

        # Apply attention weights
        weighted_sim = cos_sim * attention_weights

        return weighted_sim


class UserPreferenceModel(nn.Module):
    """Model for learning user preferences from interactions."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        self.preference_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, user_embedding: torch.Tensor,
                product_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute user preference scores for products.

        Args:
            user_embedding: Shape (embedding_dim,)
            product_embeddings: Shape (num_products, embedding_dim)

        Returns:
            preferences: Shape (num_products,)
        """
        user_exp = user_embedding.unsqueeze(
            0).expand(product_embeddings.size(0), -1)
        combined = torch.cat([user_exp, product_embeddings], dim=1)
        preferences = self.preference_encoder(combined).squeeze()

        return preferences