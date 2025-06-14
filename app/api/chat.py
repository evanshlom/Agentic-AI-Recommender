"""Chat API endpoints."""

from fastapi import APIRouter, HTTPException, Request
from app.models.api_models import ChatRequest, ChatResponse, ChatMessage, ErrorResponse
from typing import List

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Process a chat message."""
    try:
        chat_service = req.app.state.chat_service
        response = await chat_service.process_chat(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=List[ChatMessage])
async def get_chat_history(session_id: str, req: Request):
    """Get chat history for a session."""
    try:
        chat_service = req.app.state.chat_service
        history = chat_service.get_session_history(session_id)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str, req: Request):
    """Clear chat history for a session."""
    try:
        chat_service = req.app.state.chat_service
        chat_service.clear_session(session_id)
        return {"message": "Session cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))