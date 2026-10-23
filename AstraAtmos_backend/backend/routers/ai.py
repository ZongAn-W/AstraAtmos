"""
AI 解读页面 API 路由
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/ai", tags=["AI 解读"])


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    context: dict | None = None
    history: list[ChatHistoryItem] | None = None


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
async def ai_chat(request: Request, body: ChatRequest):
    """
    AI 问答接口。
    前端传入用户问题 + 可选的分析上下文 + 可选的历史对话。
    返回大模型生成的自然语言解读。
    """
    try:
        ai = request.app.state.ai_service
        history = [item.model_dump() for item in body.history] if body.history else None
        answer = await ai.chat(body.question, body.context, history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 服务错误: {e}")
