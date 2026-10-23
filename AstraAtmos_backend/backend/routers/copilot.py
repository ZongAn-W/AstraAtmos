"""
Ares Copilot API router.
Separated from /ai/chat used by AI interpretation page.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/copilot", tags=["Ares Copilot"])


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class CopilotChatRequest(BaseModel):
    question: str
    context: dict | None = None
    history: list[ChatHistoryItem] | None = None


class CopilotChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=CopilotChatResponse)
async def copilot_chat(request: Request, body: CopilotChatRequest):
    try:
        copilot = request.app.state.copilot_service
        history = [item.model_dump() for item in body.history] if body.history else None
        answer = await copilot.chat(body.question, body.context, history)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Copilot service error: {exc}")
