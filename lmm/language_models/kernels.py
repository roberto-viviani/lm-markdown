"""
Definition of language models chat kernels prompts.

This module contains a set of predefined prompt templates,

    - "summarizer"
    - "question_generator"
    - "query"
    - "query_with_context"
    - "check_content"

These prompts may be retrieved from a module-level dictionary, as
shown in the example below.

**Example**:

    ```python
    from lmm.language_models.kernels import kernel_prompts
    prompt_template: str = kernel_prompts["summarizer"]
    ```

New prompt templates may be added dynamically to the dictionary.

**Example**:

    ```python
    from lmm.language_models.kernels import (
        kernel_prompts,
        create_prompt,
    )
    create_prompt("Provide the questions the following text answers:\\n"
        + "\\nTEXT:\\n{text}", name = "question_creation")
    prompt_template: str = kernel_prompts["question_creation"]
    ```

The reason to store the prompt template in this dictionary, as shown
above, is that the prompt template is now available to other functions
in the library, like the `create_kernel` functions.
"""

from typing import Literal
from .lazy_dict import LazyLoadingDict

# Define here the kernel supported by the package.
KernelNames = Literal[
    "summarizer",
    "question_generator",
    "query",
    "query_with_context",
    "check_content",
]


# A functional returning the prompts for these kernels.
def _get_prompts(kernel_name: KernelNames) -> str:
    match kernel_name:
        case "summarizer":  # --- kernel case definition
            return """Write a concise summary of the following: "{text}"
SUMMARY:
"""
        case "question_generator":  # --- kernel case definition
            return """Provide the questions that are answered by the 
following text. To form the questions, use the format in the EXAMPLES 
OF QUESTIONS below:

EXAMPLES OF QUESTIONS:
- When is it appropriate to use logistic regression?
- What is the link function?

TEXT: "{text}"

QUESTIONS:
"""
        case "query_with_context":  # --- kernel case definition
            return """Please assist the user QUERY about the following TEXT. 
Use the CONTEXT if it helps clarifying the query, but base your response on TEXT.
----
CONTEXT: "{context}"

----
QUERY:  "{query}"

----
TEXT:  "{text}"

----
YOUR RESPONSE:

"""
        case "query":  # --- kernel case definition
            return """Please assist the user query concerning the following text:
----
QUERY:  "{query}"

----
TEXT:  "{text}"

----
YOUR RESPONSE:

"""
        case "check_content":  # --- kernel case definition
            return """Classify the category of the content of the following text. 
Examples of categories are: 'statistics', 'medicine', 'software programming', 'apology', 'human interaction', 'general knowledge'. 


Example: 
"Thank you for your explanation"
'human interaction'

TEXT CONTENT:
{text}
"""
        case _:  # do not remove this
            raise ValueError(f"Invalid kernel: {kernel_name}")


# a global object store to get the prompts
kernel_prompts = LazyLoadingDict(_get_prompts)


def create_prompt(prompt_template: str, prompt_name: str) -> None:
    """
    Adds a custom prompt template to the prompt dictionary.
    """
    kernel_prompts[prompt_name] = prompt_template  # type: ignore
