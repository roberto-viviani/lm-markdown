"""
Creates Langchain kernels.

Example:
    ```python
    from lmm.language_models.langchain.kernel import create_kernel
    model_query = create_kernel("query")
    model_summary = create_kernel("summarizer",
                                {'source': "OpenAI", 'name_model': "gpt-4o"})
    ```

Note:
    Support for new models may be added in the models module.
    Support for new kernels may be added in the kernels module in
        lmm.language_models; the create_kernel function of the present
        module must be updated to map the kernel to a language model
        type (major, minor, aux).
"""

from langchain_core.runnables.base import (
    RunnableSerializable,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from lmm.config.config import (
    Settings,
    LanguageModelSettings,
)
from .models import create_model_from_settings

from ..lazy_dict import LazyLoadingDict
from ..kernels import KernelNames, kernel_prompts, KernelModel

# for external use
KernelType = RunnableSerializable[dict[str, str], str]


# The factory function
def _create_kernel(
    model: KernelModel,
) -> RunnableSerializable[dict[str, str], str]:
    """Assembles a Langchain chain with a prompt from kernel_prompts
    and a language model as specified by a LanguageModelSettings."""

    prompt: PromptTemplate = PromptTemplate.from_template(
        kernel_prompts[model.kernel_name]
    )
    language_model: BaseChatModel = create_model_from_settings(
        model.settings
    )
    kernel = prompt | language_model | StrOutputParser()  # type: ignore
    kernel.name = f"{model.kernel_name}:{model.settings.get_model_source()}/{model.settings.get_model_name()}"
    return kernel  # type: ignore


# global project-wide repository of kernels
kernel_factory = LazyLoadingDict(_create_kernel)


# Provides the shunting of the model to the major, minor, or aux
# specification provided in config.toml
def create_kernel(
    kernel_name: KernelNames,
    user_settings: (
        dict[str, str] | LanguageModelSettings | None
    ) = None,
) -> RunnableSerializable[dict[str, str], str]:
    """
    Creates a Langchain kernel by combining configuration from config.toml
    with optional override settings. The kernel factory uses lazy loading
    to cache and reuse kernel instances based on their configuration.

    The function maps different kernel types to their appropriate language
    model settings categories. For example,
    - 'query', 'query_with_context' -> major model settings
    - 'question_generator', 'summarizer' -> minor model settings
    - 'check_content' -> aux model settings

    Settings Hierarchy (highest to lowest priority):
    1. user_settings parameter (if provided)
    2. config.toml file settings
    3. Default settings from Settings class

    Args:
        kernel_name: The name of the kernel to create. Must be one of the
            supported kernel names defined in KernelNames literal type.
        user_settings: Optional settings to override the default
            configuration. Can be either:
            - dict[str, str]: Dictionary with 'source' and 'name_model' keys
            - LanguageModelSettings: Pydantic model instance
            - None: Use settings from config.toml or defaults

    Returns:
        RunnableSerializable[dict[str, str], str]: A Langchain runnable
        chain that combines a prompt template, language model, and string
        output parser. The chain accepts a dictionary of template variables
        and returns a string response.

    Raises:
        ValueError: If kernel_name is not supported or if user_settings
            contains invalid model source names. No check is made at this
            stage that the model names are correct (as they frequently
            change); instead, failure occurs when the .invoke member function
            is called.

    Examples:
        Create kernel with default settings:
        ```python
        >>> kernel = create_kernel("query")
        ```

        Override with dictionary:
        ```python
        >>> kernel = create_kernel("summarizer",
        ...     {"source": "OpenAI", "name_model": "gpt-4o"})
        ```

        Override with settings object:
        ```python
        >>> settings = LanguageModelSettings(source="Mistral",
        ...     name_model="mistral-small-latest")
        >>> kernel = create_kernel("question_generator", settings)
        ```
    """
    match kernel_name:
        case 'query' | 'query_with_context':
            if bool(user_settings):  # disallow empty dictionary
                settings = Settings(major=user_settings)  # type: ignore
            else:
                settings = Settings()
            return kernel_factory[
                KernelModel(
                    kernel_name=kernel_name,
                    settings=settings.major,
                )
            ]
        case 'question_generator' | 'summarizer':
            if bool(user_settings):  # disallow empty dictionary
                settings = Settings(minor=user_settings)  # type: ignore
            else:
                settings = Settings()
            return kernel_factory[
                KernelModel(
                    kernel_name=kernel_name,
                    settings=settings.minor,
                )
            ]
        case 'check_content':
            if bool(user_settings):  # disallow empty dictionary
                settings = Settings(aux=user_settings)  # type: ignore
            else:
                settings = Settings()
            return kernel_factory[
                KernelModel(
                    kernel_name=kernel_name,
                    settings=settings.aux,
                )
            ]
        case _:
            # unreachable code if match on mernel names is exhaustive
            if bool(user_settings):  # disallow empty dictionary
                settings = Settings(minor=user_settings)  # type: ignore
            else:
                settings = Settings()
            return kernel_factory[
                KernelModel(
                    kernel_name=kernel_name,
                    settings=settings.minor,
                )
            ]
