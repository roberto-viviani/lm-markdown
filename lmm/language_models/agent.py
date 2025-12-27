"""
Agent abstraction that combines a model, a prompt, and tools.
"""

from typing import Any

from lmm.language_models.base import BaseChatModel
from lmm.language_models.messages import Message
from lmm.language_models.prompts import PromptDefinition


class Agent:
    """
    An agent wraps a language model with a specific prompt and optional tools.
    
    It manages the conversation history and the tool execution loop (future).
    """

    def __init__(
        self, 
        model: BaseChatModel, 
        prompt: str, 
        system_prompt: str | None = None,
        tools: list[PromptDefinition] | None = None,
        name: str | None = None,
    ):
        self.model = model
        self.prompt_template = prompt
        self.system_prompt = system_prompt
        self.tools = tools
        self.name = name
        self.history: list[Message] = []

    def get_name(self) -> str:
        """Return the name of the agent."""
        return self.name or "Agent"

    def invoke(self, input_data: str | dict[str, Any]) -> str:
        """
        Synchronous invocation of the agent.

        Args:
            input_data: Input string or dictionary to format the prompt.

        Returns:
            The agent's response text.
        """
        # 1. Format prompt
        if isinstance(input_data, str):
            formatted_prompt = self.prompt_template.format(text=input_data)
        else:
            formatted_prompt = self.prompt_template.format(**input_data)

        # 2. Construct messages
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role='system', content=self.system_prompt))
        
        # Add history (not fully implemented in this MVP, but placeholder)
        # messages.extend(self.history)
        
        messages.append(Message(role='user', content=formatted_prompt))

        # 3. Call model
        response = self.model.chat(messages, self.tools)

        # 4. Handle response (simple text return for now)
        if response.content:
            return response.content
        return ""

    # Future: Add chat() method for stateful interaction
