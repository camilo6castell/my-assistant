from __future__ import annotations

from pydantic import BaseModel, Field


class DemoQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    collections: list[str] = Field(default_factory=list)
    mode: str = Field(default="SOFT", pattern="^(SOFT|HARD)$")
