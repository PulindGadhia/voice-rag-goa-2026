from fastapi import APIRouter, Request

from ..schemas.requests import TextQueryRequest
from ..schemas.responses import QueryResponse

router = APIRouter()


@router.post("/api/query", response_model=QueryResponse)
async def query(request: Request, payload: TextQueryRequest) -> QueryResponse:
    orchestrator = request.app.state.orchestrator
    return await orchestrator.process_text_query(payload)

