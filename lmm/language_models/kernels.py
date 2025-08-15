"""
Definition of language models kernels and respective prompts.

Example:
    ```python
    from lmm.language_models.kernels import kernel_prompts
    prompt_template: str = kernel_promts["summarizer"]
"""

from typing import Literal
from .lazy_dict import LazyLoadingDict

# Define here the kernel defined in the package.
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
        case "summarizer":
            return """Write a concise summary of the following: "{text}"
SUMMARY:
"""
        case "question_generator":
            return """Provide 1-{n} questions that are answered by the 
following text. Use the format in the EXAMPLE below:

EXAMPLE:
- When is it appropriate to use logistic regression?
- What is the link function?

TEXT: "{text}"

QUESTIONS:
"""
        case "query_with_context":
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
        case "query":
            return """Please assist the user query concerning the following text:
----
QUERY:  "{query}"

----
TEXT:  "{text}"

----
YOUR RESPONSE:

"""
        case "check_content":
            return """Classify the category of the content of the following text. 
Examples of categories are: 'statistics', 'medicine', 'software programming', 'apology', 'human interaction', 'general knowledge'. 


Example: 
"Thank you for your explanation"
'human interaction'

TEXT CONTENT:
{text}
"""
        case _:
            raise ValueError(f"Invalid kernel: {kernel_name}")


kernel_prompts = LazyLoadingDict(_get_prompts)
