"""
sessions_router.py

HTTP layer only — translates requests/responses. All actual logic
(budgeting, eviction, persistence, completion) lives in the collaborators
it's handed; this file just wires them together per-request.
"""

from fastapi import APIRouter, Depends, HTTPException

from ..auth import verify_api_key
from ..models.schemas import (
    NewSessionRequest, NewSessionResponse, ChatRequest, ChatResponse, SessionInfo,
)
from ..sessions.store import SessionStore
from ..llm.completion_client import CompletionClient


def build_sessions_router(store: SessionStore, completion_client: CompletionClient) -> APIRouter:
    router = APIRouter(dependencies=[Depends(verify_api_key)])

    @router.post("/sessions", response_model=NewSessionResponse)
    def create_session(req: NewSessionRequest):
        s = store.create_session(req.device_name, req.system_prompt)
        return NewSessionResponse(session_id=s.session_id)

    @router.get("/sessions", response_model=list[SessionInfo])
    def list_sessions():
        return [
            SessionInfo(session_id=s.session_id, device_name=s.device_name,
                        turns=len(s.history), last_active=s.last_active)
            for s in store.list_sessions()
        ]

    @router.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        store.delete(session_id)
        return {"deleted": session_id}

    @router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
    def chat(session_id: str, req: ChatRequest):
        if store.get(session_id) is None:
            raise HTTPException(status_code=404, detail="unknown session_id")

        store.add_turn(session_id, "user", req.message)
        messages = store.build_messages(session_id)
        reply = completion_client.complete(messages, req.max_tokens, req.temperature)
        store.add_turn(session_id, "assistant", reply)

        return ChatResponse(session_id=session_id, reply=reply)

    return router
