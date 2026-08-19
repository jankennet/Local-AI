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


class NewSessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.7


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class SessionInfo(BaseModel):
    session_id: str
    device_name: str
    turns: int
    last_active: float
