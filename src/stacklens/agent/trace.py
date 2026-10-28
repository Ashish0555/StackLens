"""Rendering the agent's internal conversation as a readable trail."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

_PREVIEW_LENGTH = 100


def format_trace(messages: list[BaseMessage]) -> str:
    """Turn the raw message list into a line-per-step reasoning trail."""
    return "\n".join(line for message in messages for line in _format_message(message))


def _format_message(message: BaseMessage) -> list[str]:
    if isinstance(message, HumanMessage):
        return [f"[VOCÊ] {message.content}"]

    if isinstance(message, AIMessage) and message.tool_calls:
        return [
            f"[AGENTE decide] chamar {call['name']}({call['args']})" for call in message.tool_calls
        ]

    if isinstance(message, ToolMessage):
        return [f"[FERRAMENTA {message.name}] -> {_preview(str(message.content))}"]

    if isinstance(message, AIMessage) and message.content:
        return [f"[AGENTE responde] {message.content}"]

    return []


def _preview(content: str) -> str:
    """Tool output can be long; a preview keeps the trail readable."""
    single_line = content[:_PREVIEW_LENGTH].replace("\n", " ")
    return f"{single_line}..." if len(content) > _PREVIEW_LENGTH else single_line
