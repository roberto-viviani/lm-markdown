"""
This module centralizes storage and definition of resources to
configure a language model to perform a specific function ('kernel').

This consists at present of a set of prompts that specialize the
function of a chat with a language model. These data are collected in
a ToolDefinition object, which identifies the kernels uniquely. The
module supports predefined toolset definitions and the specification
of custom definitions to interoperate with the rest of the library.

The predefined prompt sets are

    - "summarizer"
    - "question_generator"
    - "query"
    - "query_with_context"
    - "check_content"

These prompts may be retrieved from the module-level dictionary
`tool_library`, as shown in the example below.

**Example**:

    ```python
    from lmm.language_models.tools import tool_library
    prompt_template: str = tool_library["summarizer"]
    ```

Note that the prompt template text may be to be adapted to the
framework to be used. For example, for Langchain,

    ```python
    from langchain_core.prompts import PromptTemplate
    prompt: PromptTemplate = PromptTemplate.from_template(
        kernel_prompts["summarizer"]
    )
    ```

New prompt text templates may be added dynamically to the dictionary.
To do this, one can use the `create_prompt` function.

**Example**:

    ```python
    from lmm.language_models.tools import (
        tool_library,
        create_prompt,
    )
    create_prompt("Provide the questions the following text answers:\\n"
        + "\\nTEXT:\\n{text}", name = "question_creation")
    prompt_template: str = tool_library["question_creation"]
    ```

There is no much added value in storing the prompt template text in
the library per se. The motivation is that the prompt becomes a tool
that is now available to other functions in the library, like the
`create_runnable` function (see lmm.language_models.langchain.runnables).
A runnable with the prompt can then be obtained directly, e.g.

    ```python
    from lmm.language_models.langchain.runnables import create_runnable
    lmm = create_runnable("question_creation") # langchain runnable
    response = lmm.invoke({'text': "Apples are healthy food"})
    ```

Note that the name of the runnable kernel is that of the tool set (here
a set of prompts) that defines the kernal uniquely.
"""

from typing import Literal
from pydantic import BaseModel, ConfigDict
from .lazy_dict import LazyLoadingDict


class ToolDefinition(BaseModel):
    """Groups all properties that uniquely define a kernel tool"""

    name: str
    prompt: str
    system_prompt: str | None = None

    model_config = ConfigDict(frozen=False, extra='forbid')


# List of pre-defined kernel tools
KernelNames = Literal[
    "summarizer",
    "question_generator",
    "query",
    "query_with_context",
    "check_content",
]


# A functional returning the tool definitions. This is used for
# the typed dictionary containing the prompt texts.
def _create_tool(kernel_name: KernelNames) -> ToolDefinition:
    match kernel_name:
        case "summarizer":  # --- kernel case definition
            return ToolDefinition(
                name=kernel_name,
                prompt="""
Write a concise summary of the following: "{text}"

SUMMARY:
""",
            )
        case "question_generator":  # --- kernel case definition
            return ToolDefinition(
                name=kernel_name,
                prompt="""
Provide at most five questions that are answered by the following 
text, focussing on important or general issues. To form the 
questions, use the format in the EXAMPLES OF QUESTIONS below:

EXAMPLES OF QUESTIONS:
- When is it appropriate to use logistic regression?
- What is the link function?

TEXT: "{text}"

QUESTIONS:
""",
                system_prompt="You are a helpful teacher.",
            )
        case "query_with_context":  # --- kernel case definition
            return ToolDefinition(
                name=kernel_name,
                prompt="""
Please assist the user QUERY about the following TEXT. 
Use the CONTEXT if it helps clarifying the query, but base your response on TEXT.
----
CONTEXT: "{context}"

----
QUERY:  "{query}"

----
TEXT:  "{text}"

----
YOUR RESPONSE:

""",
                system_prompt="You are a helpful assistant.",
            )
        case "query":  # --- kernel case definition
            return ToolDefinition(
                name=kernel_name,
                prompt="""
Please assist the user and answert the query concerning the following text:
----
QUERY:  "{query}"

----
TEXT:  "{text}"

----
YOUR RESPONSE:

""",
            )
        case "check_content":  # --- kernel case definition
            return ToolDefinition(
                name=kernel_name,
                prompt="""
Classify the category of the content of the following text. 
Examples of categories are: 'statistics', 'medicine', 'software programming', 'apology', 'human interaction', 'general knowledge'. 


Example: 
"Thank you for your explanation"
'human interaction'

TEXT CONTENT:
{text}
""",
            )
        case _:  # do not remove this
            raise ValueError(f"Invalid kernel: {kernel_name}")


# a module-level typed dictionary for the preformed prompts
tool_library = LazyLoadingDict(_create_tool)


def create_prompt(
    prompt: str,
    name: str,
    *,
    system_prompt: str | None = None,
) -> None:
    """
    Adds a custom prompt template to the prompt dictionary.

    Args:
        prompt: the prompt text.
        name: the name of the tool. This will also define a kernel
            with the same name (i.e. a Langchain runnable)
        system_prompt: an optional system prompt text.
    """

    # We abuse the lack of run-time checks for Literals here. We do
    # this because we want the availability of the preformed prompts
    # given by Literal but also the flexibility to add new prompts.
    definition = ToolDefinition(
        name=name,
        prompt=prompt,
        system_prompt=system_prompt,
    )
    tool_library[name] = definition  # type: ignore
