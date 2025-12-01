"""
Abstract base class for language model backends.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, AsyncIterator
import asyncio

from lmm.config.config import LanguageModelSettings
from lmm.language_models.messages import Message
from lmm.language_models.prompts import ToolDefinition


class BaseChatModel(ABC):
    """Abstract base class for chat models."""

    def __init__(self, settings: LanguageModelSettings):
        self.settings = settings

    @abstractmethod
    def chat(
        self, 
        messages: list[Message], 
        tools: list[ToolDefinition] | None = None
    ) -> Message:
        """
        Get the next message from the model synchronously.
        
        Args:
            messages: The conversation history.
            tools: Optional list of tools the model can call.
            
        Returns:
            The model's response (which may contain text content or tool calls).
        """
        pass

    async def achat(
        self, 
        messages: list[Message], 
        tools: list[ToolDefinition] | None = None
    ) -> Message:
        """
        Get the next message from the model asynchronously.
        
        Default implementation delegates to the synchronous chat method
        in a thread pool.
        """
        return await asyncio.to_thread(self.chat, messages, tools)

    def stream(
        self, 
        messages: list[Message], 
        tools: list[ToolDefinition] | None = None
    ) -> Iterator[str]:
        """
        Stream the model's response synchronously.
        
        Default implementation yields the full response content at once.
        Note: This default implementation does not support streaming tool calls.
        """
        response = self.chat(messages, tools)
        if response.content:
            yield response.content

    async def astream(
        self, 
        messages: list[Message], 
        tools: list[ToolDefinition] | None = None
    ) -> AsyncIterator[str]:
        """
        Stream the model's response asynchronously.
        
        Default implementation yields the full response content at once.
        """
        response = await self.achat(messages, tools)
        if response.content:
            yield response.content
