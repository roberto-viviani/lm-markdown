"""
Creates Langchain kernel objects. These objects may be used in
Langchain chains.

This module currently supports chat kernel objects. The objects
are based on the predefined templates from the kernel_prompts
dictionary in the lmm.language_models.prompts module.

In langchain, objects are invoked with a dictionary that contains
the parameters for the prompt template.

Example of a predefined kernel:
    ```python
    from lmm.language_models.langchain.kernel import create_kernel
    model_query = create_kernel("query")  # uses config.toml for model
    model_questions = create_kernel("question_generator",
                                {'model': "OpenAI/gpt-4o"})
    response = model_questions.invoke({
        'text': "Logistic regression is typically used when the "
            + "outcome variable is binary."
    })
    ```

Example of a dynamically created chat kernel:
    ```python
        from lmm.language_models.kernels import (
            kernel_prompts,
            create_prompt,
        )

        prompt_template = '''Provide the questions to which the text answers.
            TEXT:
            {text}
        '''
        create_prompt(prompt_template, "question_generator")

        # create a kernel from the major model in config.toml with
        # this prompts
        settings = Settings()
        model = create_kernel("question_generator", settings.major)

        # if no settings object given, defaults to settings.minor
        model_minor = create_kernel("question_generator")
    ```
"""

from pydantic import BaseModel, ConfigDict

from langchain_core.runnables.base import (
    RunnableSerializable,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

from lmm.config.config import (
    Settings,
    LanguageModelSettings,
    EmbeddingSettings,
)
from .models import (
    create_model_from_settings,
    create_embedding_model_from_settings,
)

from ..lazy_dict import LazyLoadingDict
from ..prompts import KernelNames, kernel_prompts


# Defines the kernel/model combinations. We need a class here
# to use it as a key in a lazy_dict
class KernelModel(BaseModel):
    """Kernel definition"""

    kernel_name: KernelNames
    settings: LanguageModelSettings

    model_config = ConfigDict(frozen=True, extra='forbid')


# Defines the embedding model (we use the settings directly)
EmbeddingModel = EmbeddingSettings

# Exports the type of the Langchain object
KernelType = RunnableSerializable[dict[str, str], str]


# The factory functions
def _create_kernel(
    model: KernelModel,
) -> KernelType:  # RunnableSerializable[dict[str, str], str]
    """Assembles a Langchain chain with a prompt from kernel_prompts
    and a language model as specified by a LanguageModelSettings."""

    prompt: PromptTemplate = PromptTemplate.from_template(
        kernel_prompts[model.kernel_name]
    )
    language_model: BaseChatModel = create_model_from_settings(
        model.settings
    )
    kernel: KernelType = prompt | language_model | StrOutputParser()  # type: ignore
    # .name is a member function of RunnableSerializable
    # inited to None, which we initialize here
    kernel.name = f"{model.kernel_name}:{model.settings.get_model_source()}/{model.settings.get_model_name()}"
    return kernel


def _create_embedding(
    settings: EmbeddingSettings,
) -> Embeddings:
    """Assembles a Langchain embedding object from specs"""
    return create_embedding_model_from_settings(settings)


# global project-wide repository of kernels
kernel_factory = LazyLoadingDict(_create_kernel)
embeddings_factory = LazyLoadingDict(_create_embedding)


# Provides the shunting of the model to the major, minor, or aux
# specification provided in config.toml
def create_kernel(
    kernel_name: KernelNames | str,
    user_settings: (
        dict[str, str] | LanguageModelSettings | None
    ) = None,
) -> KernelType:  # RunnableSerializable[dict[str, str], str]
    """
    Creates a Langchain kernel by combining configuration from
    config.toml with optional override settings. The kernel factory
    uses lazy loading to cache and reuse kernel instances based on
    their configuration.

    The function maps different kernel types to their appropriate
    language model settings categories. For example,
    - 'query', 'query_with_context' -> major model settings
    - 'question_generator', 'summarizer' -> minor model settings
    - 'check_content' -> aux model settings

    Settings Hierarchy (highest to lowest priority):
    1. user_settings parameter (if provided)
    2. config.toml file settings
    3. Default settings from Settings class

    Args:
        kernel_name: The name of the kernel to create. If one of
            the supported kernel names defined in the KernelNames
            literal type, returns a cached kernel object. Otherwise,
            looks up in the kernel_prompts dictionary if there is
            a prompt with that kernel_name, and returns a kernel
            object for a chat with that prompt.
        user_settings: Optional settings to override the default
            configuration. Can be either:
            - dict[str, str]: Dictionary with 'model' key
            - LanguageModelSettings: Pydantic model instance
            - None: Use settings from config.toml or defaults

    Returns:
        RunnableSerializable[dict[str, str], str]: A Langchain
        runnable chain that combines a prompt template, language
        model, and string output parser. The chain accepts a
        dictionary of template variables and returns a string
        response.

    Raises:
        ValueError: If kernel_name is not supported or if user_settings
            contains invalid model source names. No check is made at this
            stage that the model names are correct (as they frequently
            change); instead, failure occurs when the .invoke member function
            is called.
        ValidationError, TypeError: alternative errors raised in the same
            circumstances as above.
        ImportError: for not installed libraries.

    Examples:
        Create kernel with default settings read from configuration
        file:
        ```python
        kernel = create_kernel("query")
        ```

        Override with dictionary:
        ```python
        kernel = create_kernel("summarizer",
            {"model": "OpenAI/gpt-4o"})
        ```

        Override with settings object:
        ```python
        from lmm.config.config import LanguageModelSettings
        settings = LanguageModelSettings(
            model="Mistral/mistral-small-latest"
        )
        kernel = create_kernel("question_generator", settings)
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
        case str():
            # allows custom chat kernels to be constructed from custom
            # prompt templates
            if bool(user_settings):  # disallow empty dictionary
                settings = Settings(minor=user_settings)  # type: ignore
            else:
                settings = Settings()
            if kernel_name not in kernel_prompts.keys():
                raise ValueError(
                    f"Invalid kernel name: {kernel_name}"
                )
            prompt: PromptTemplate = PromptTemplate.from_template(
                kernel_prompts[kernel_name]  # type: ignore
            )

            kernel = create_kernel_from_objects(
                prompt,
                settings.minor,
            )
            kernel.name = f"{kernel_name}:{settings.minor}"
            return kernel
        case _:
            raise ValueError("Unreacheable code reached.")


def create_embeddings(
    settings: dict[str, str] | EmbeddingSettings | None = None,
) -> Embeddings:
    """
    Creates a Langchain embeddings kernel from a configuration
    object.

    Args:
        settings: an EmbeddingSettings object with the following
            fields:

            - dense_model: a specification in the form provider/
            model, for example 'OpenAI/text-embedding-3-small'
            - sparse_model: a sparse model specification.

            Alternatively, a dictionary with the same fields and
            text. If None (default), the settings will be read
            from the configuration file. If no configuration file
            exists, a settings object will be created with default
            parameters.

    Returns:
        a Langchain object that embeds text by calling embed_documents
            or embed_query.

    Raises:
        ValidationError, TypeError: for invalid spec
        ImportError: for missing libraries
        requests.ConnectionError: if not online
    """
    if not bool(settings):  # includes empty dict
        sets = Settings()
        settings = sets.embeddings
    elif isinstance(settings, dict):
        # checked by pydantic model
        settings = EmbeddingSettings(**settings)  # type: ignore

    return embeddings_factory[settings]


def create_kernel_from_objects(
    prompt: PromptTemplate,
    language_model: (
        BaseChatModel | LanguageModelSettings | None
    ) = None,
) -> KernelType:
    """
    Creates a Langchain runnable from a prompt template and
    a language settings object.

    Args:
        prompt: a prompt template object
        language_model: either a Langchain BaseChatModel, or
            a LanguageModelSettings object, or None (default). In
            this latter case the language.minor from the config
            file is used to create the model.

    Returns:
        a Langchain runnable, a type aliased as `KernelType`.

    """
    if language_model is None:
        settings = Settings()
        language_model = create_model_from_settings(settings.minor)
        name = f"Custom:{settings.minor}"
    elif isinstance(language_model, LanguageModelSettings):
        name = f"Custom:{language_model}"
        language_model = create_model_from_settings(language_model)
    else:  # it's a BaseChatModel
        name = "Custom"

    kernel: KernelType = (  # type: ignore
        prompt | language_model | StrOutputParser()
    )
    kernel.name = name
    return kernel
