"""
Generic data structures for language model interactions.
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Represents a tool call requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    """Represents a message in a chat conversation."""
    role: Literal['system', 'user', 'assistant', 'tool']
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = Field(
        default=None, 
        description="ID of the tool call this message responds to"
    )
