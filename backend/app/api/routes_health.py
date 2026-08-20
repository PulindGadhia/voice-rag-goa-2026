from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    orchestrator = request.app.state.orchestrator
    providers = orchestrator.get_provider_status() if hasattr(orchestrator, "get_provider_status") else {}
    return {"status": "ok", "ready": orchestrator.is_ready, "providers": providers}


@router.get("/ready")
async def ready(request: Request) -> dict:
    orchestrator = request.app.state.orchestrator
    if not orchestrator.is_ready:
        return {"status": "not_ready", "ready": False}
    details = await orchestrator.retrieval_engine.health_check()
    providers = orchestrator.get_provider_status() if hasattr(orchestrator, "get_provider_status") else {}
    return {"status": "ready", "ready": True, "providers": providers, **details}


@router.get("/warmup")
@router.post("/warmup")
async def warmup(request: Request) -> dict:
    orchestrator = request.app.state.orchestrator
    warmup_res = await orchestrator.warmup() if hasattr(orchestrator, "warmup") else {}
    providers = orchestrator.get_provider_status() if hasattr(orchestrator, "get_provider_status") else {}
    return {"status": "warmed", "ready": orchestrator.is_ready, "providers": providers, **warmup_res}


