from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/api/metrics")
async def metrics(request: Request) -> dict:
    return request.app.state.orchestrator.metrics.snapshot()

