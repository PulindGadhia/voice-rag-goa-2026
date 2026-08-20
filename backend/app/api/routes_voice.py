from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

router = APIRouter()


@router.post("/api/voice/query")
async def voice_query(
    request: Request,
    audio: UploadFile = File(...),
    language: str = Form("en"),
    top_k: int | None = Form(None),
) -> object:
    data = await audio.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio file exceeds 10 MB")
    if not data:
        raise HTTPException(status_code=400, detail="audio file is empty")
    return await request.app.state.orchestrator.process_voice_query(
        data,
        content_type=audio.content_type or "audio/wav",
        language=language,
        top_k=top_k,
    )

