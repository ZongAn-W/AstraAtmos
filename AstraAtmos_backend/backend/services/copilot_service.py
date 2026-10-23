"""
Ares Copilot service.
Separated from AI interpretation page service to avoid prompt/fallback coupling.
"""

from __future__ import annotations

import logging
from typing import Any

from services.ai_service import AIService, PLAIN_TEXT_PROMPT

logger = logging.getLogger("astraatmos.copilot")

COPILOT_SYSTEM_PROMPT = """你是 Ares Copilot，专门解读“数据总览”页面当前图表的实时快照。
请遵守：
1. 使用中文回答，2-3 句话，尽量控制在 80-180 字。
2. 先说明图表在表达什么，再给出一个关键现象或变量关系。
3. 如果图表数据未就绪，不要编造，直接提示“数据尚未就绪”并给出下一步操作建议。
4. 不要要求用户跳转到“预测分析”页面。
"""


class CopilotService(AIService):
    """Ares Copilot chat service for Data Overview page."""

    async def chat(
        self,
        question: str,
        context: dict | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        question = (question or "").strip()
        if not question:
            return "请先点击“AI 解读当前图表”，我会基于当前卡片快照给出简要解释。"

        if not self.api_key:
            return self._normalize_plain_text(self._builtin_reply(question, context))

        messages: list[dict[str, str]] = [
            {"role": "system", "content": COPILOT_SYSTEM_PROMPT},
            {"role": "system", "content": PLAIN_TEXT_PROMPT},
        ]

        if context:
            messages.append(
                {
                    "role": "system",
                    "content": f"当前图表上下文：\n{self._format_copilot_context(context)}",
                }
            )

        sanitized_history = self._sanitize_history(history)
        if sanitized_history:
            messages.extend(sanitized_history)

        messages.append({"role": "user", "content": question})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 700,
            "temperature": 0.3,
        }

        try:
            response = await self.client.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            answer = self._extract_answer_text(data)
            if answer:
                return self._normalize_plain_text(answer)
            return self._normalize_plain_text(self._builtin_reply(question, context))
        except Exception as exc:
            logger.error("Copilot API call failed: %s", exc)
            return self._normalize_plain_text(self._builtin_reply(question, context))

    def _format_copilot_context(self, context: dict[str, Any]) -> str:
        parts: list[str] = []
        card = context.get("expanded_card_title") or context.get("expanded_card") or "unknown"
        parts.append(f"card: {card}")

        ls_range = context.get("ls_range")
        if ls_range is not None:
            parts.append(f"ls_range: {ls_range}")

        selected_variables = context.get("selected_variables")
        if isinstance(selected_variables, list) and selected_variables:
            parts.append(f"selected_variables: {selected_variables}")

        active_mode = context.get("active_mode")
        if active_mode:
            parts.append(f"active_mode: {active_mode}")

        coord = context.get("coordinate")
        if coord:
            parts.append(f"coordinate: {coord}")

        dynamic_metrics = context.get("dynamic_metrics")
        if dynamic_metrics:
            text = str(dynamic_metrics)
            if len(text) > 2600:
                text = text[:2600] + "\n...[TRUNCATED]"
            parts.append(f"dynamic_metrics:\n{text}")

        card_snapshot = context.get("card_snapshot")
        if card_snapshot:
            snapshot_text = str(card_snapshot)
            if len(snapshot_text) > 2200:
                snapshot_text = snapshot_text[:2200] + "\n...[TRUNCATED]"
            parts.append(f"card_snapshot:\n{snapshot_text}")

        return "\n".join(parts)

    def _builtin_reply(self, question: str, context: dict | None) -> str:
        del question
        if not context:
            return "当前图表上下文为空。请先展开一个图表卡片，再点击“AI 解读当前图表”。"

        dynamic_metrics = context.get("dynamic_metrics")
        card_snapshot = context.get("card_snapshot")
        has_data = bool(dynamic_metrics) or bool(card_snapshot)

        if not has_data:
            return "当前图表数据尚未就绪。建议等待图表加载完成，或切换到其他已渲染的卡片后重试。"

        card = context.get("expanded_card_title") or context.get("expanded_card") or "当前图表"
        return f"{card} 的快照已读取。建议继续观察同一 Ls 下不同变量的联动变化，以确认当前异常是局地波动还是全局趋势。"
