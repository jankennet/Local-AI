"""
schemas.py

All request/response shapes in one place. Routers import from here
instead of defining inline models, so the API contract is easy to find
and easy to version later.
"""

from typing import Optional
from pydantic import BaseModel


class NewSessionRequest(BaseModel):
    device_name: str = "unknown device"
    system_prompt: Optional[str] = None
    external_id: Optional[str] = None
    metadata: Optional[dict] = None


class NewSessionResponse(BaseModel):
    session_id: str
    external_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7
    use_rag: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    context_used: int
    context_limit: int


class SessionInfo(BaseModel):
    session_id: str
    device_name: str
    turns: int
    last_active: float
    context_used: int
    context_limit: int
    external_id: Optional[str] = None
    metadata: Optional[dict] = None
    source: Optional[str] = None


class SessionDetailResponse(BaseModel):
    session_id: str
    device_name: str
    system_prompt: str
    history: list
    summary: str
    created_at: float
    last_active: float
    external_id: Optional[str] = None
    metadata: Optional[dict] = None
    source: Optional[str] = None
    context_used: int
    context_limit: int