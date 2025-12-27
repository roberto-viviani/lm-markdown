"""
LangChain adapter implementation.
"""

from collections.abc import Iterator, AsyncIterator

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)

# from lmm.config.config import LanguageModelSettings
from lmm.language_models.base import BaseChatModel
from lmm.language_models.messages import Message  # , ToolCall
from lmm.language_models.prompts import PromptDefinition
from lmm.language_models.langchain.models import create_model_from_settings


class LangChainChatModel(BaseChatModel):
    """Adapter for LangChain chat models."""

    def _convert_messages(self, messages: list[Message]) -> list[BaseMessage]:
        """Convert generic messages to LangChain messages."""
        lc_messages: list[BaseMessage] = []
        for msg in messages:
            if msg.role == 'system':
                lc_messages.append(SystemMessage(content=msg.content or ""))
            elif msg.role == 'user':
                lc_messages.append(HumanMessage(content=msg.content or ""))
            elif msg.role == 'assistant':
                # TODO: Handle tool calls in assistant message
                lc_messages.append(AIMessage(content=msg.content or ""))
            elif msg.role == 'tool':
                lc_messages.append(
                    ToolMessage(
                        content=msg.content or "", 
                        tool_call_id=msg.tool_call_id or ""
                    )
                )
        return lc_messages

    def _convert_response(self, response: BaseMessage) -> Message:
        """Convert LangChain response to generic message."""
        content: str | None = str(response.content) if response.content else None # type: ignore
        # TODO: Extract tool calls from response.additional_kwargs or tool_calls
        return Message(role='assistant', content=content)

    def chat(
        self, 
        messages: list[Message], 
        tools: list[PromptDefinition] | None = None
    ) -> Message:
        lc_messages = self._convert_messages(messages)
        model = create_model_from_settings(self.settings)
        
        # TODO: Bind tools if provided
        
        response = model.invoke(lc_messages)
        return self._convert_response(response)

    async def achat(
        self, 
        messages: list[Message], 
        tools: list[PromptDefinition] | None = None
    ) -> Message:
        lc_messages = self._convert_messages(messages)
        model = create_model_from_settings(self.settings)
        
        response = await model.ainvoke(lc_messages)
        return self._convert_response(response)

    def stream(
        self, 
        messages: list[Message], 
        tools: list[PromptDefinition] | None = None
    ) -> Iterator[str]:
        lc_messages = self._convert_messages(messages)
        model = create_model_from_settings(self.settings)
        
        for chunk in model.stream(lc_messages):
            if isinstance(chunk, str):
                yield chunk
            elif hasattr(chunk, 'content'):
                yield str(chunk.content)  # type: ignore (langchain type)

    async def astream(
        self, 
        messages: list[Message], 
        tools: list[PromptDefinition] | None = None
    ) -> AsyncIterator[str]:
        lc_messages = self._convert_messages(messages)
        model = create_model_from_settings(self.settings)
        
        async for chunk in model.astream(lc_messages):
            if isinstance(chunk, str):
                yield chunk
            elif hasattr(chunk, 'content'):
                yield str(chunk.content)  # type: ignore (langchain type)
