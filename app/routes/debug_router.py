"""
debug_router.py

Development/debug endpoints for vector store inspection.
NOT for production — disable via LLM_ENABLE_DEBUG=false
"""

import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import verify_api_key
from ..embeddings import EmbeddingService, VectorStore, create_vector_store


class VectorSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str | None = None


class VectorSearchResponse(BaseModel):
    results: list[dict]
    total_vectors: int


class VectorAddRequest(BaseModel):
    text: str
    metadata: dict = {}


class VectorStatsResponse(BaseModel):
    backend: str
    collection: str
    path: str
    vector_count: int
    embedding_model: str
    embedding_dim: int


# Module-level cache for vector store instances (per backend+path+collection)
_vector_store_cache: dict[str, VectorStore] = {}


def _get_vector_store(
    embedding_service: EmbeddingService,
    vector_backend: str,
    vector_db_path: str,
    vector_collection: str,
) -> VectorStore:
    """Get or create cached vector store instance."""
    key = f"{vector_backend}:{vector_db_path}:{vector_collection}"
    if key not in _vector_store_cache:
        _vector_store_cache[key] = create_vector_store(
            embedding_service,
            backend=vector_backend,
            path=vector_db_path,
            collection_name=vector_collection,
        )
    return _vector_store_cache[key]


def build_debug_router(
    store,  # SessionStore (avoid circular import)
    embedding_service: EmbeddingService,
    vector_backend: str,
    vector_db_path: str,
    vector_collection: str,
) -> APIRouter:
    router = APIRouter(dependencies=[Depends(verify_api_key)])
    enabled = os.environ.get("LLM_ENABLE_DEBUG", "false").lower() == "true"

    def _check_enabled():
        if not enabled:
            raise HTTPException(status_code=404, detail="Debug endpoints disabled. Set LLM_ENABLE_DEBUG=true")

    @router.get("/debug/vector/stats", response_model=VectorStatsResponse)
    def vector_stats():
        _check_enabled()
        vs = _get_vector_store(embedding_service, vector_backend, vector_db_path, vector_collection)
        return VectorStatsResponse(
            backend=vector_backend,
            collection=vector_collection,
            path=vector_db_path,
            vector_count=len(vs),
            embedding_model=embedding_service.model_name,
            embedding_dim=embedding_service.dimension,
        )

    @router.post("/debug/vector/search", response_model=VectorSearchResponse)
    def vector_search(req: VectorSearchRequest):
        _check_enabled()
        vs = _get_vector_store(embedding_service, vector_backend, vector_db_path, vector_collection)

        filter_ = None
        if req.session_id and vector_backend == "qdrant":
            from qdrant_client.http import models as qmodels
            filter_ = qmodels.Filter(
                must=[qmodels.FieldCondition(key="session", match=qmodels.MatchValue(value=req.session_id))]
            )

        # SimpleVectorStore doesn't support filter
        if filter_:
            try:
                results = vs.search(req.query, top_k=req.top_k, filter=filter_)
            except TypeError:
                results = vs.search(req.query, top_k=req.top_k)
        else:
            results = vs.search(req.query, top_k=req.top_k)

        return VectorSearchResponse(
            results=[
                {"score": score, "text": text, "metadata": meta}
                for score, text, meta in results
            ],
            total_vectors=len(vs),
        )

    @router.post("/debug/vector/add")
    def vector_add(req: VectorAddRequest):
        _check_enabled()
        vs = _get_vector_store(embedding_service, vector_backend, vector_db_path, vector_collection)
        point_id = vs.add(req.text, req.metadata)
        return {"id": point_id, "total_vectors": len(vs)}

    @router.delete("/debug/vector/clear")
    def vector_clear():
        _check_enabled()
        vs = _get_vector_store(embedding_service, vector_backend, vector_db_path, vector_collection)
        vs.clear()
        return {"cleared": True, "total_vectors": len(vs)}

    @router.get("/debug/sessions/{session_id}/vectors")
    def session_vectors(session_id: str):
        _check_enabled()
        s = store.get(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "turns": len(s.history),
            "vector_store_type": type(s._vector_store).__name__ if s._vector_store else "none",
            "history": s.history,
        }

    @router.get("/debug/sessions/{session_id}/tokens")
    def session_tokens(session_id: str):
        _check_enabled()
        s = store.get(session_id)
        if not s:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Use the store's internal token counter
        breakdown = s.get_token_breakdown(store._counter)
        
        return {
            "session_id": session_id,
            "total_tokens": breakdown["total"],
            "breakdown": breakdown,
            "budget": store.budget,
            "available": max(0, store.budget - breakdown["total"]),
            "utilization_pct": round(breakdown["total"] / store.budget * 100, 1) if store.budget > 0 else 0,
        }

    return router