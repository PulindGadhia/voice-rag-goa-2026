from pydantic import BaseModel, Field


class TextQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    language: str = Field(default="en", max_length=20)

