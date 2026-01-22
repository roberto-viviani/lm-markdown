"""
This module centralizes storage and definition of prompts to
configure language models to perform a specific function. 

These prompts are collected in a PromptDefinition object, which 
identifies the prompt set (system and user) uniquely, and specifies
the language model tier that should be used with the prompt. The 
module supports predefined prompts as well as prompts that may be 
dynamically added to the centralized repository.

The predefined prompt sets are

    - "summarizer"
    - "question_generator"
    - "query"
    - "query_with_context"
    - "context_validator"
    - "allowed_content_validator"

These prompts may be retrieved from the module-level dictionary
`prompt_library`, as shown in the example below.

**Example**:

    ```python
    from lmm.language_models.prompts import (
        prompt_library, 
        PromptDefinition,
    )
    prompt_definition: PromptDefinition = prompt_library["summarizer"]
    ```

New prompt text templates may be added dynamically to the dictionary.
To do this, one can use the `create_prompt` function.

**Example**:

    ```python
    from lmm.language_models.prompts import (
        prompt_library,
        create_prompt,
    )
    create_prompt("Provide the questions the following text answers:\\n"
        + "\\nTEXT:\\n{text}", name = "question_creation")
    prompt_template: str = prompt_library["question_creation"]
    ```

There is no much added value in storing the prompt template text in
the library per se. The motivation is that the prompt becomes a tool
that is now available to other functions in the library, like the
`create_runnable` function (see lmm.language_models.langchain.runnables).
A runnable with the prompt can then be obtained by requesting it with
the name of the prompt object:

    ```python
    from lmm.language_models.langchain.runnables import create_runnable
    lmm = create_runnable("question_creation") # langchain runnable
    response = lmm.invoke({'text': "Apples are healthy food"})
    ```
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict
from .lazy_dict import LazyLoadingDict

ModelTier = Literal['major', 'minor', 'aux']


class PromptDefinition(BaseModel):
    """Groups all properties that uniquely define a kernel tool"""

    name: str
    prompt: str
    system_prompt: str | None = None
    model_tier: ModelTier = 'minor'

    model_config = ConfigDict(frozen=False, extra='forbid')


# List of pre-defined prompts
PromptNames = Literal[
    "summarizer",
    "chat_summarizer",
    "question_generator",
    "query",
    "query_with_context",
    "context_validator",
    "allowed_content_validator",
]


# A functional returning the prompt definitions. This is the factory
# function that creates the prompt object (an objects containing the
# prompts).
def create_prompt_definition(
    prompt_name: PromptNames, **kwargs: object | list[object]
) -> PromptDefinition:
    match prompt_name:
        case "summarizer":  # --- kernel case definition
            return PromptDefinition(
                name=prompt_name,
                prompt="""
Write a concise summary of the following: "{text}"

SUMMARY:
""",
                model_tier='minor',
            )
        case "chat_summarizer": # --- kernel case definition
            return PromptDefinition(
                name=prompt_name,
                prompt="""
You find below, separated by '###', a conversation and a follow-up 
question to the conversation. Use the conversation to provide the 
context of the question to facilitate retrieval of relevant material
from a vector database. Do NOT respond to the query, do NOT provide a
comprehensive summary of the conversation, provide ONLY the context
that is relevant to understand the new query.

###
CONVERSATION: "{text}"

###
QUESTION: "{query}"

###
CONTEXT:
""",
                model_tier="aux",
            )
        case "question_generator":  # --- kernel case definition
            return PromptDefinition(
                name=prompt_name,
                prompt="""
Provide at most five questions that are answered by the following 
text, focussing on important or general issues. Use the format in the 
EXAMPLES OF QUESTIONS below:

EXAMPLES OF QUESTIONS:
- When is it appropriate to use logistic regression? - What is the link function?

TEXT: {text}

QUESTIONS:
""",
                system_prompt="You are a helpful teacher.",
                model_tier='minor',
            )
        case "query_with_context":  # --- kernel case definition
            return PromptDefinition(
                name=prompt_name,
                prompt="""
Please assist the user QUERY about the following TEXT. 
Use the CONTEXT if it helps clarifying the query, but base your response on TEXT.
----
CONTEXT: {context}

----
QUERY:  {query}

----
TEXT:  {text}

----
YOUR RESPONSE:

""",
                system_prompt="You are a helpful assistant.",
                model_tier='major',
            )
        case "query":  # --- kernel case definition
            return PromptDefinition(
                name=prompt_name,
                prompt="""
Please assist the user and answer the query concerning the following text:
----
QUERY:  {query}

----
TEXT:  {text}

----
YOUR RESPONSE:

""",
                model_tier='major',
            )
        case "context_validator":  # --- kernel case definition
            return PromptDefinition(
                name=prompt_name,
                prompt="""
Your task is to evaluate if the context information provided is relevant to a query and its response.
You have two options to answer. Either YES or NO.
Answer YES, if the context information is relevant for the query and response, otherwise NO.

----
Query and Response: {query}

----
Context: {context}

----
Answer:
""",
                system_prompt="You are a helpful assistant.",
                model_tier='aux',
            )
        case "allowed_content_validator":  # --- kernel case def.
            param: object | list[object] = kwargs.pop(
                'allowed_content', None
            )
            match param:
                case list():
                    try:
                        allowed_content: list[str] = [
                            f"'{c}'" for c in param  # type: ignore
                        ]
                        allowed_content = list(
                            set(allowed_content)
                        )  # unique
                    except Exception as e:
                        raise ValueError(
                            "Kernel allowed_content_validator: invalid provider_param"
                        ) from e
                case str():
                    allowed_content = [param]
                case None:
                    allowed_content = []
                case _:
                    raise ValueError(
                        "Invalid allowed_content_validator kernel spec: "
                        f"invalid provider_param: {param}"
                    )

            if not (allowed_content):
                raise ValueError(
                    "Invalid allowed_content_validator kernel spec: "
                    "missing provider_param"
                )

            content_list: str = ", ".join(allowed_content)
            return PromptDefinition(
                name=prompt_name,
                prompt="""
Classify the following text into one of the following categories: 
"""
                + content_list
                + """, 'apology', 'human interaction', 'general knowledge'.

Output: only the category

TEXT CONTENT:
{text}

RESPONSE:
""",
                model_tier='aux',
                system_prompt="You are a helpful assistant.",
            )
        case _:  # do not remove this
            raise ValueError(f"Invalid kernel: {prompt_name}")


# a module-level typed dictionary for the preformed prompts
prompt_library = LazyLoadingDict(create_prompt_definition)


def create_prompt(
    prompt: str,
    name: str,
    *,
    system_prompt: str | None = None,
    replace: bool = False,
) -> None:
    """
    Adds a custom prompt template to the prompt dictionary.

    Args:
        prompt: the prompt text.
        name: the name of the tool. This will also define a kernel
            with the same name (i.e. a Langchain runnable)
        system_prompt: an optional system prompt text.
    """

    if name in prompt_library.keys():
        if replace:
            prompt_library.pop(name)  # type: ignore
        else:
            raise ValueError(
                f"'{name}' is already a registered prompt. "
                "Use another name to register a custom prompt."
            )

    # We abuse the lack of run-time checks for Literals here. We do
    # this because we want the availability of the preformed prompts
    # given by Literal but also the flexibility to add new prompts.
    definition = PromptDefinition(
        name=name,
        prompt=prompt,
        system_prompt=system_prompt,
    )
    prompt_library[name] = definition  # type: ignore
