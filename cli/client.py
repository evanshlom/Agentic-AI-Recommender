"""HTTP client for the FastAPI service."""

import httpx
from typing import Dict, Any, List, Optional
import os


class ApiClient:
    """Client for interacting with the FastAPI service."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or f"http://{os.getenv('API_HOST', 'localhost')}:{os.getenv('API_PORT', '8000')}"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def send_message(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a chat message."""
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
            
        response = await self.client.post(
            f"{self.base_url}/api/chat/",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def get_products(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get products with optional filters."""
        params = filters or {}
        response = await self.client.get(
            f"{self.base_url}/api/products/",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def get_recommendations(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recommendations for a session."""
        response = await self.client.post(
            f"{self.base_url}/api/recommendations/",
            json={"session_id": session_id, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        response = await self.client.get(f"{self.base_url}/api/recommendations/graph/stats")
        response.raise_for_status()
        return response.json()