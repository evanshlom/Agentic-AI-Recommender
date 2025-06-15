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

        # Get edge types from metadata
        edge_types = metadata[1] if len(metadata) > 1 else []

        # First heterogeneous convolution layer - include all edge types
        conv1_dict = {}
        for edge_type in edge_types:
            source, relation, target = edge_type
            is_bipartite = source != target
            conv1_dict[edge_type] = GATConv(
                hidden_dim, hidden_dim, heads=num_heads, concat=False,
                add_self_loops=not is_bipartite
            )

        # Add default edge types if not in metadata
        default_edges = [
            ('product', 'similar_to', 'product'),
            ('product', 'belongs_to', 'category'),
            ('product', 'has_style', 'style'),
            ('user', 'prefers', 'product'),
            ('user', 'viewed', 'product'),
        ]

        for edge_type in default_edges:
            if edge_type not in conv1_dict:
                source, relation, target = edge_type
                is_bipartite = source != target
                conv1_dict[edge_type] = GATConv(
                    hidden_dim, hidden_dim, heads=num_heads, concat=False,
                    add_self_loops=not is_bipartite
                )

        self.conv1 = HeteroConv(conv1_dict, aggr='mean')

        # Second heterogeneous convolution layer - subset of edge types
        conv2_dict = {}
        for edge_type in edge_types:
            source, relation, target = edge_type
            is_bipartite = source != target
            conv2_dict[edge_type] = GATConv(
                hidden_dim, out_dim, heads=1, concat=False,
                add_self_loops=not is_bipartite
            )

        # Add some default second layer edges if empty
        if not conv2_dict:
            conv2_dict = {
                ('product', 'similar_to', 'product'): GATConv(
                    hidden_dim, out_dim, heads=1, concat=False, add_self_loops=True
                ),
                ('user', 'prefers', 'product'): GATConv(
                    hidden_dim, out_dim, heads=1, concat=False, add_self_loops=False
                ),
            }

        self.conv2 = HeteroConv(conv2_dict, aggr='mean')

        # Dimension projection layers for nodes not updated by conv2
        self.dim_projections = nn.ModuleDict()
        for node_type in node_types:
            self.dim_projections[node_type] = Linear(hidden_dim, out_dim)

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
        try:
            conv1_out = self.conv1(h_dict, edge_index_dict)
            # Update h_dict with conv1 outputs, keeping originals for missing nodes
            for node_type in h_dict:
                if node_type in conv1_out:
                    h_dict[node_type] = F.relu(conv1_out[node_type])
                # If not updated by conv1, keep the original projection
        except Exception as e:
            # Fallback: use original projections if convolution fails
            pass

        # Filter edge_index_dict to only include edges that exist in conv2
        filtered_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type in self.conv2.convs:
                filtered_edge_index_dict[edge_type] = edge_index

        # Second convolution layer with filtered edges
        conv2_updated = set()
        try:
            if filtered_edge_index_dict:
                conv2_out = self.conv2(h_dict, filtered_edge_index_dict)
                # Update h_dict with conv2 outputs and track which nodes were updated
                for node_type in h_dict:
                    if node_type in conv2_out:
                        h_dict[node_type] = conv2_out[node_type]
                        conv2_updated.add(node_type)
        except Exception as e:
            # If conv2 fails, we'll use the conv1 outputs
            pass

        # Project nodes that weren't updated by conv2 from hidden_dim to out_dim
        for node_type in h_dict:
            if node_type not in conv2_updated and node_type in self.dim_projections:
                # Node wasn't updated by conv2, so it's still at hidden_dim
                # Project it to out_dim
                h_dict[node_type] = self.dim_projections[node_type](
                    h_dict[node_type])

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
