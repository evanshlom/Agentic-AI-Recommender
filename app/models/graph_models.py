"""Graph data models for the ecommerce chatbot."""

from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import torch
from torch_geometric.data import HeteroData
import numpy as np


class NodeType(str, Enum):
    """Types of nodes in our graph."""
    USER = "user"
    PRODUCT = "product"
    CATEGORY = "category"
    STYLE = "style"
    BRAND = "brand"
    COLOR = "color"


class EdgeType(str, Enum):
    """Types of edges in our graph."""
    SIMILAR_TO = "similar_to"
    BELONGS_TO = "belongs_to"
    HAS_STYLE = "has_style"
    MADE_BY = "made_by"
    HAS_COLOR = "has_color"
    PREFERS = "prefers"
    VIEWED = "viewed"
    PURCHASED = "purchased"


class BaseNode(BaseModel):
    """Base node structure."""
    id: str
    type: NodeType
    features: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @field_validator('embedding')
    def validate_embedding(cls, v):
        if v is not None and len(v) != 128:
            raise ValueError("Embedding must be 128-dimensional")
        return v


class ProductNode(BaseNode):
    """Product node with specific attributes."""
    type: NodeType = NodeType.PRODUCT
    name: str
    category: str
    style: str
    price: float
    colors: List[str]
    sizes: List[str]
    brand: str
    rating: float = Field(ge=0, le=5)
    description: str = ""

    def to_feature_vector(self) -> List[float]:
        """Convert product to feature vector for GNN."""
        # Simple feature encoding
        features = [
            self.price / 100.0,  # Normalize price
            self.rating / 5.0,   # Normalize rating
            len(self.colors) / 10.0,
            len(self.sizes) / 10.0,
        ]
        return features


class UserNode(BaseNode):
    """User node with preferences."""
    type: NodeType = NodeType.USER
    session_id: str
    preferences: Dict[str, float] = Field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list)

    def update_preference(self, key: str, value: float):
        """Update a specific preference."""
        self.preferences[key] = value


class GraphEdge(BaseModel):
    """Edge between nodes."""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EcommerceGraph:
    """Main graph structure for the ecommerce system."""

    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}
        self.edges: List[GraphEdge] = []
        self._adjacency: Dict[str, Dict[str, List[GraphEdge]]] = {}

    def add_node(self, node: BaseNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = {}

    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph."""
        self.edges.append(edge)

        # Update adjacency list
        if edge.source not in self._adjacency:
            self._adjacency[edge.source] = {}
        if edge.edge_type.value not in self._adjacency[edge.source]:
            self._adjacency[edge.source][edge.edge_type.value] = []
        self._adjacency[edge.source][edge.edge_type.value].append(edge)

    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[BaseNode]:
        """Get neighboring nodes of a specific type."""
        if node_id not in self._adjacency:
            return []

        neighbors = []
        if edge_type:
            edges = self._adjacency[node_id].get(edge_type.value, [])
            for edge in edges:
                if edge.target in self.nodes:
                    neighbors.append(self.nodes[edge.target])
        else:
            for edge_list in self._adjacency[node_id].values():
                for edge in edge_list:
                    if edge.target in self.nodes:
                        neighbors.append(self.nodes[edge.target])

        return neighbors

    def get_products_by_category(self, category: str) -> List[ProductNode]:
        """Get all products in a category."""
        products = []
        for node in self.nodes.values():
            if isinstance(node, ProductNode) and node.category == category:
                products.append(node)
        return products

    def get_similar_products(self, product_id: str, limit: int = 5) -> List[Tuple[ProductNode, float]]:
        """Get similar products with similarity scores."""
        similar = []
        edges = self._adjacency.get(product_id, {}).get(
            EdgeType.SIMILAR_TO.value, [])

        for edge in edges[:limit]:
            if edge.target in self.nodes:
                product = self.nodes[edge.target]
                if isinstance(product, ProductNode):
                    similar.append((product, edge.weight))

        return sorted(similar, key=lambda x: x[1], reverse=True)

    def to_hetero_data(self) -> HeteroData:
        """Convert graph to PyTorch Geometric HeteroData."""
        data = HeteroData()

        # Group nodes by type
        node_groups = {}
        node_id_map = {}

        for node in self.nodes.values():
            if node.type not in node_groups:
                node_groups[node.type] = []
                node_id_map[node.type] = {}

            idx = len(node_groups[node.type])
            node_id_map[node.type][node.id] = idx
            node_groups[node.type].append(node)

        # Add node features
        for node_type, nodes in node_groups.items():
            if nodes:
                if node_type == NodeType.PRODUCT:
                    # Create feature matrix for products
                    features = []
                    for node in nodes:
                        features.append(node.to_feature_vector())
                    data[node_type.value].x = torch.tensor(
                        features, dtype=torch.float)
                else:
                    # Simple placeholder features for other node types
                    data[node_type.value].x = torch.randn(len(nodes), 4)

        # Add edges
        edge_groups = {}
        for edge in self.edges:
            # Get node types
            source_node = self.nodes.get(edge.source)
            target_node = self.nodes.get(edge.target)

            if source_node and target_node:
                edge_key = (source_node.type.value,
                            edge.edge_type.value, target_node.type.value)

                if edge_key not in edge_groups:
                    edge_groups[edge_key] = {'indices': [], 'weights': []}

                source_idx = node_id_map[source_node.type][edge.source]
                target_idx = node_id_map[target_node.type][edge.target]

                edge_groups[edge_key]['indices'].append(
                    [source_idx, target_idx])
                edge_groups[edge_key]['weights'].append(edge.weight)

        # Add edge indices
        for edge_key, edge_data in edge_groups.items():
            if edge_data['indices']:
                indices = torch.tensor(
                    edge_data['indices'], dtype=torch.long).t()
                data[edge_key].edge_index = indices
                data[edge_key].edge_attr = torch.tensor(
                    edge_data['weights'], dtype=torch.float).unsqueeze(1)

        return data

    def create_subgraph(self, node_ids: Set[str], max_hops: int = 2) -> 'EcommerceGraph':
        """Create a subgraph containing specified nodes and their neighborhoods."""
        subgraph = EcommerceGraph()
        visited = set()
        to_visit = list(node_ids)

        for _ in range(max_hops + 1):
            next_visit = []
            for node_id in to_visit:
                if node_id in visited or node_id not in self.nodes:
                    continue

                visited.add(node_id)
                subgraph.add_node(self.nodes[node_id])

                # Add edges
                for edge_type_edges in self._adjacency.get(node_id, {}).values():
                    for edge in edge_type_edges:
                        if edge.target not in visited:
                            next_visit.append(edge.target)
                        subgraph.add_edge(edge)

            to_visit = next_visit

        return subgraph

    def calculate_similarity(self, node1_id: str, node2_id: str) -> float:
        """Calculate similarity between two nodes."""
        node1 = self.nodes.get(node1_id)
        node2 = self.nodes.get(node2_id)

        if not node1 or not node2:
            return 0.0

        if node1.type != node2.type:
            return 0.0

        # Simple similarity based on shared neighbors
        neighbors1 = set(n.id for n in self.get_neighbors(node1_id))
        neighbors2 = set(n.id for n in self.get_neighbors(node2_id))

        if not neighbors1 and not neighbors2:
            return 0.0

        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)

        return intersection / union if union > 0 else 0.0
