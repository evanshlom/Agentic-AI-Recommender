"""Graph service for managing the ecommerce graph."""

import json
from typing import Dict, List, Optional, Any
from app.models.graph_models import (
    EcommerceGraph, ProductNode, UserNode, GraphEdge,
    NodeType, EdgeType
)
from app.models.api_models import GraphStats


class GraphService:
    """Service for managing the ecommerce graph."""

    def __init__(self):
        self.graph = EcommerceGraph()
        self.sessions: Dict[str, UserNode] = {}
        self._initialize_sample_data()

    def _initialize_sample_data(self):
        """Initialize the graph with sample product data."""
        # Sample products
        products = [
            # Business shirts
            ProductNode(
                id="prod-001",
                name="Oxford Business Shirt",
                category="shirts",
                style="business-casual",
                price=79.99,
                colors=["white", "light-blue", "navy"],
                sizes=["S", "M", "L", "XL"],
                brand="ProWear",
                rating=4.5,
                description="Classic Oxford shirt perfect for business casual"
            ),
            ProductNode(
                id="prod-002",
                name="Modern Fit Dress Shirt",
                category="shirts",
                style="business",
                price=89.99,
                colors=["white", "gray", "black"],
                sizes=["S", "M", "L", "XL"],
                brand="ExecutiveStyle",
                rating=4.7,
                description="Sleek modern fit for professional settings"
            ),

            # Casual shirts
            ProductNode(
                id="prod-003",
                name="Weekend Henley",
                category="shirts",
                style="casual",
                price=45.99,
                colors=["navy", "charcoal", "olive"],
                sizes=["S", "M", "L", "XL", "XXL"],
                brand="ComfortWear",
                rating=4.3,
                description="Comfortable henley for casual weekends"
            ),
            ProductNode(
                id="prod-004",
                name="Casual Linen Shirt",
                category="shirts",
                style="casual",
                price=59.99,
                colors=["white", "beige", "light-blue"],
                sizes=["M", "L", "XL"],
                brand="SummerStyle",
                rating=4.4,
                description="Breathable linen perfect for warm weather"
            ),

            # Pants
            ProductNode(
                id="prod-005",
                name="Chino Business Pants",
                category="pants",
                style="business-casual",
                price=95.99,
                colors=["khaki", "navy", "gray"],
                sizes=["30x30", "32x30", "32x32", "34x32"],
                brand="ProWear",
                rating=4.6,
                description="Versatile chinos for office and after"
            ),
            ProductNode(
                id="prod-006",
                name="Comfort Stretch Jeans",
                category="pants",
                style="casual",
                price=79.99,
                colors=["dark-blue", "black"],
                sizes=["30x30", "32x30", "32x32", "34x32", "36x32"],
                brand="DenimCo",
                rating=4.8,
                description="Premium denim with comfortable stretch"
            ),

            # Dresses
            ProductNode(
                id="prod-007",
                name="Professional Sheath Dress",
                category="dresses",
                style="business",
                price=129.99,
                colors=["black", "navy", "burgundy"],
                sizes=["S", "M", "L"],
                brand="ExecutiveStyle",
                rating=4.7,
                description="Elegant sheath dress for the office"
            ),
            ProductNode(
                id="prod-008",
                name="Casual Summer Dress",
                category="dresses",
                style="casual",
                price=69.99,
                colors=["floral", "blue", "coral"],
                sizes=["S", "M", "L", "XL"],
                brand="SummerStyle",
                rating=4.5,
                description="Light and breezy for summer days"
            ),

            # Jackets
            ProductNode(
                id="prod-009",
                name="Classic Blazer",
                category="jackets",
                style="business",
                price=199.99,
                colors=["navy", "charcoal", "black"],
                sizes=["S", "M", "L", "XL"],
                brand="ExecutiveStyle",
                rating=4.8,
                description="Timeless blazer for professional settings"
            ),
            ProductNode(
                id="prod-010",
                name="Casual Bomber Jacket",
                category="jackets",
                style="casual",
                price=89.99,
                colors=["olive", "black", "navy"],
                sizes=["M", "L", "XL"],
                brand="StreetStyle",
                rating=4.4,
                description="Trendy bomber for casual outings"
            ),

            # Shoes
            ProductNode(
                id="prod-011",
                name="Oxford Dress Shoes",
                category="shoes",
                style="business",
                price=149.99,
                colors=["black", "brown"],
                sizes=["8", "9", "10", "11", "12"],
                brand="ProWear",
                rating=4.6,
                description="Classic oxfords for formal occasions"
            ),
            ProductNode(
                id="prod-012",
                name="Casual Sneakers",
                category="shoes",
                style="casual",
                price=79.99,
                colors=["white", "black", "gray"],
                sizes=["8", "9", "10", "11", "12"],
                brand="ComfortWear",
                rating=4.5,
                description="Comfortable sneakers for everyday wear"
            ),

            # Athletic wear
            ProductNode(
                id="prod-013",
                name="Performance Polo",
                category="shirts",
                style="athletic",
                price=54.99,
                colors=["black", "navy", "gray"],
                sizes=["S", "M", "L", "XL"],
                brand="SportTech",
                rating=4.4,
                description="Moisture-wicking polo for active days"
            ),
            ProductNode(
                id="prod-014",
                name="Athletic Shorts",
                category="pants",
                style="athletic",
                price=39.99,
                colors=["black", "navy", "gray"],
                sizes=["S", "M", "L", "XL"],
                brand="SportTech",
                rating=4.3,
                description="Lightweight shorts for workouts"
            )
        ]

        # Add products to graph
        for product in products:
            self.graph.add_node(product)

        # Create similarity edges
        similarities = [
            ("prod-001", "prod-002", 0.8),  # Business shirts
            ("prod-003", "prod-004", 0.85),  # Casual shirts
            ("prod-005", "prod-006", 0.6),   # Different pant styles
            ("prod-007", "prod-009", 0.7),   # Business dress and blazer
            ("prod-011", "prod-001", 0.6),   # Dress shoes and business shirt
            ("prod-013", "prod-014", 0.9),   # Athletic items
            ("prod-001", "prod-005", 0.7),   # Business casual combo
            ("prod-003", "prod-006", 0.75),  # Casual combo
        ]

        for source, target, weight in similarities:
            self.graph.add_edge(GraphEdge(
                source=source,
                target=target,
                edge_type=EdgeType.SIMILAR_TO,
                weight=weight
            ))
            # Add reverse edge for bidirectional similarity
            self.graph.add_edge(GraphEdge(
                source=target,
                target=source,
                edge_type=EdgeType.SIMILAR_TO,
                weight=weight
            ))

    def get_or_create_user(self, session_id: str) -> UserNode:
        """Get or create a user node for a session."""
        user_id = f"user_{session_id}"

        if user_id not in self.graph.nodes:
            user = UserNode(
                id=user_id,
                session_id=session_id,
                preferences={},
                interaction_history=[]
            )
            self.graph.add_node(user)
            self.sessions[session_id] = user

        return self.graph.nodes[user_id]

    def get_user_node(self, session_id: str) -> Optional[UserNode]:
        """Get user node by session ID."""
        user_id = f"user_{session_id}"
        node = self.graph.nodes.get(user_id)
        return node if isinstance(node, UserNode) else None

    def add_interaction(self, user_id: str, product_id: str, interaction_type: str):
        """Add a user-product interaction."""
        if interaction_type == "viewed":
            edge_type = EdgeType.VIEWED
        elif interaction_type == "purchased":
            edge_type = EdgeType.PURCHASED
        else:
            edge_type = EdgeType.PREFERS

        self.graph.add_edge(GraphEdge(
            source=user_id,
            target=product_id,
            edge_type=edge_type,
            weight=1.0
        ))

    def get_all_products(self, filters: Optional[Dict[str, Any]] = None) -> List[ProductNode]:
        """Get all products with optional filters."""
        products = []

        for node in self.graph.nodes.values():
            if isinstance(node, ProductNode):
                # Apply filters
                if filters:
                    if filters.get("category") and node.category != filters["category"]:
                        continue
                    if filters.get("style") and node.style != filters["style"]:
                        continue
                    if filters.get("min_price") and node.price < filters["min_price"]:
                        continue
                    if filters.get("max_price") and node.price > filters["max_price"]:
                        continue
                    if filters.get("min_rating") and node.rating < filters["min_rating"]:
                        continue
                    if filters.get("colors"):
                        if not any(color in node.colors for color in filters["colors"]):
                            continue
                    if filters.get("brands") and node.brand not in filters["brands"]:
                        continue

                products.append(node)

        return products

    def get_graph_stats(self) -> GraphStats:
        """Get statistics about the graph."""
        node_counts = {}
        edge_counts = {}

        # Count nodes by type
        for node in self.graph.nodes.values():
            node_type = node.type.value
            node_counts[node_type] = node_counts.get(node_type, 0) + 1

        # Count edges by type
        for edge in self.graph.edges:
            edge_type = edge.edge_type.value
            edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1

        return GraphStats(
            total_nodes=len(self.graph.nodes),
            total_edges=len(self.graph.edges),
            node_counts=node_counts,
            edge_counts=edge_counts,
            active_sessions=len(self.sessions)
        )
