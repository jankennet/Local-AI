"""
sessions_router.py

HTTP layer only — translates requests/responses. All actual logic
(budgeting, eviction, persistence, completion, tool-calling) lives in
the collaborators it's handed; this file just wires them together
per-request.
"""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ..auth import verify_api_key, verify_api_key_ws
from ..models.schemas import (
    NewSessionRequest, NewSessionResponse, ChatRequest, ChatResponse, SessionInfo,
    SessionDetailResponse,
)
from ..sessions.store import SessionStore
from ..llm.completion_client import CompletionClient
from ..llm.agent_loop import run_agent_turn
from ..llm.agent_loop_streaming import run_agent_turn_streaming, StreamEvent
from ..llm.agents import get_orchestrator, AgentType
from ..metrics import (
    record_session_created, record_session_deleted, record_session_expired,
    set_session_tokens, remove_session_metrics,
)
from ..config import settings

logger = logging.getLogger(__name__)


def build_sessions_router(
    store: SessionStore,
    completion_client: CompletionClient,
    tools: dict,
    tool_timeout_seconds: float = 30.0,
    tool_max_retries: int = 2,
) -> APIRouter:
    router = APIRouter(dependencies=[Depends(verify_api_key)])

    @router.post("/sessions", response_model=NewSessionResponse)
    def create_session(req: NewSessionRequest):
        try:
            s = store.create_session(
                req.device_name,
                req.system_prompt,
                external_id=req.external_id or "",
                metadata=req.metadata,
            )
            record_session_created()
            set_session_tokens(s.session_id, 0, store.budget)
            return NewSessionResponse(session_id=s.session_id, external_id=s.external_id or None)
        except Exception as e:
            logger.exception("Failed to create session: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/sessions", response_model=list[SessionInfo])
    def list_sessions(user_id: str = None, source: str = None, external_id: str = None):
        if external_id:
            s = store.get_by_external_id(external_id)
            sessions = [s] if s else []
        elif user_id:
            sessions = store.list_by_user(user_id)
        elif source:
            sessions = store.list_by_source(source)
        else:
            sessions = store.list_sessions()

        return [
            SessionInfo(
                session_id=s.session_id,
                device_name=s.device_name,
                turns=len(s.history),
                last_active=s.last_active,
                context_used=store.tokens_used(s.session_id),
                context_limit=store.budget,
                external_id=s.external_id or None,
                metadata=s.metadata or None,
                source=s.external_id.split(":")[0] if s.external_id else None,
            )
            for s in sessions
        ]

    @router.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        store.delete(session_id)
        record_session_deleted()
        remove_session_metrics(session_id)
        return {"deleted": session_id}

    @router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
    def get_session(session_id: str):
        s = store.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="unknown session_id")
        return SessionDetailResponse(
            session_id=s.session_id,
            device_name=s.device_name,
            system_prompt=s.system_prompt,
            history=s.history,
            summary=s.summary,
            created_at=s.created_at,
            last_active=s.last_active,
            external_id=s.external_id or None,
            metadata=s.metadata or None,
            source=s.external_id.split(":")[0] if s.external_id else None,
            context_used=store.tokens_used(s.session_id),
            context_limit=store.budget,
        )

    @router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
    async def chat(session_id: str, req: ChatRequest):
        if store.get(session_id) is None:
            raise HTTPException(status_code=404, detail="unknown session_id")

        try:
            store.add_turn(session_id, "user", req.message)
        except Exception as e:
            logger.exception("Failed to add turn: %s", e)
            raise HTTPException(status_code=500, detail=f"add_turn failed: {e}")

        # Use orchestrator if enabled, otherwise fall back to legacy agent_loop
        if settings.orchestrator_enabled:
            try:
                orchestrator = get_orchestrator()
                orchestrator.configure(
                    enable_planning=settings.orchestrator_planning,
                    enable_review=settings.orchestrator_review,
                )

                force_agent = None
                if settings.orchestrator_force_agent:
                    try:
                        force_agent = AgentType(settings.orchestrator_force_agent.lower())
                    except ValueError:
                        pass  # Invalid agent type, use auto-classification

                from ..llm.agents import AgentContext
                context = AgentContext(
                    session_id=session_id,
                    query=req.message,
                    store=store,
                    completion_client=completion_client,
                    tools=tools,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    rag_top_k=settings.rag_top_k,
                    rag_initial_k=settings.rag_initial_k,
                    use_reranker=settings.reranker_enabled,
                    use_rag=req.use_rag,
                    tool_timeout=tool_timeout_seconds,
                    max_retries=tool_max_retries,
                    # Budget-aware: use session store's budget as the token budget
                    token_budget=store.budget,
                    max_tool_calls=20,
                    max_rounds=12,
                )

                result = await orchestrator.execute(context, force_agent=force_agent)
                reply = result.final_reply
            except Exception as e:
                logger.exception("Orchestrator failed: %s", e)
                raise HTTPException(status_code=500, detail=f"Orchestrator failed: {e}")

            # Update session token metrics
            set_session_tokens(session_id, store.tokens_used(session_id), store.budget)

            return ChatResponse(
                session_id=session_id,
                reply=reply,
                context_used=store.tokens_used(session_id),
                context_limit=store.budget,
            )
        else:
            # Legacy path
            reply = await run_agent_turn(
                store,
                session_id,
                completion_client,
                tools,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                rag_query=req.message if req.use_rag else None,
                tool_timeout=tool_timeout_seconds,
                max_retries=tool_max_retries,
                rag_top_k=settings.rag_top_k,
                rag_initial_k=settings.rag_initial_k,
                use_reranker=settings.reranker_enabled,
            )

            # Update session token metrics
            set_session_tokens(session_id, store.tokens_used(session_id), store.budget)

            return ChatResponse(
                session_id=session_id,
                reply=reply,
                context_used=store.tokens_used(session_id),
                context_limit=store.budget,
            )

    @router.websocket("/sessions/{session_id}/chat/stream")
    async def chat_stream_ws(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for streaming agent responses."""
        # Verify API key from query params or headers
        api_key = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")
        if not api_key or not verify_api_key_ws(api_key):
            await websocket.close(code=4001, reason="Invalid API key")
            return
        
        await websocket.accept()
        
        if store.get(session_id) is None:
            await websocket.send_json({"type": "error", "data": {"message": "unknown session_id"}})
            await websocket.close()
            return

        try:
            # Receive the initial message
            data = await websocket.receive_json()
            message = data.get("message", "")
            max_tokens = data.get("max_tokens", 512)
            temperature = data.get("temperature", 0.7)
            use_rag = data.get("use_rag", False)
            
            if not message:
                await websocket.send_json({"type": "error", "data": {"message": "empty message"}})
                return

            store.add_turn(session_id, "user", message)

            async for event in run_agent_turn_streaming(
                store,
                session_id,
                completion_client,
                tools,
                max_tokens=max_tokens,
                temperature=temperature,
                rag_query=message if use_rag else None,
                tool_timeout=tool_timeout_seconds,
                max_retries=tool_max_retries,
                rag_top_k=settings.rag_top_k,
                rag_initial_k=settings.rag_initial_k,
                use_reranker=settings.reranker_enabled,
            ):
                await websocket.send_json(json.loads(event.to_json()))
                
                if event.type == "done":
                    break
            
            # Update session token metrics
            set_session_tokens(session_id, store.tokens_used(session_id), store.budget)
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error for session {session_id}: {e}")
            try:
                await websocket.send_json({"type": "error", "data": {"message": str(e)}})
            except:
                pass
        finally:
            try:
                await websocket.close()
            except:
                pass

    @router.get("/sessions/{session_id}/chat/stream")
    async def chat_stream_sse(
        session_id: str,
        message: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_rag: bool = False,
        api_key: str = Depends(verify_api_key),
    ):
        """SSE endpoint for streaming agent responses."""
        if store.get(session_id) is None:
            raise HTTPException(status_code=404, detail="unknown session_id")

        store.add_turn(session_id, "user", message)

        async def event_generator():
            try:
                async for event in run_agent_turn_streaming(
                    store,
                    session_id,
                    completion_client,
                    tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    rag_query=message if use_rag else None,
                    tool_timeout=tool_timeout_seconds,
                    max_retries=tool_max_retries,
                    rag_top_k=settings.rag_top_k,
                    rag_initial_k=settings.rag_initial_k,
                    use_reranker=settings.reranker_enabled,
                ):
                    yield {"event": event.type, "data": event.to_json()}
                    
                    if event.type == "done":
                        break
                
                # Update session token metrics
                set_session_tokens(session_id, store.tokens_used(session_id), store.budget)
            except Exception as e:
                logger.error(f"SSE error for session {session_id}: {e}")
                yield {"event": "error", "data": json.dumps({"type": "error", "data": {"message": str(e)}})}

        return EventSourceResponse(event_generator())

    return router