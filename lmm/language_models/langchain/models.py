"""
This module implements creation of Langchain language model objects
wrapping the message exchange calls of the primitive model API, thus
allowing a unified programming interface to diverse models,
abstracting from vendor details. The model objects are stored in 
two global repositories, langchain_models and langchain_embeddings.

The model objects are the lowest among the abstractions offered by
LangChain/LangGraph, but they also implement the runnable interface,
so that they can be called directly through .invoke/.ainvoke.

The creation of these objects connects the LangChain API to 
configuration settings that may be given programmatically or loaded
automatically from config.toml. The settings are given as a 
LanguageModelSettings object, a dictionary with the appropriate fields
or a spec given as argument to the create_model_from_spec function. 
The LanguageModelSettings is also a member of the Settings object that
is read from the config file prividing a default that is loaded from 
the config.toml file.

Note: the abstract class in the LangChain API that defines the model
object is `BaseChatModel`.

Examples:

```python
from lmm.language_models.langchain.models import (
    create_model_from_spec,
    create_model_from_settings,
    langchain_models,
)
from lmm.config.config import LanguageModelSettings
from langchain.chat_models import BaseChatModel

# Method 1: Using a LanguageModelSettings object
settings = LanguageModelSettings(
    model="OpenAI/gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    max_retries=3
)
model: BaseChatModel = langchain_models[settings]
# (this method accesses the repository directly).

# Method 2: Using create_model_from_settings function
model = create_model_from_settings(settings)
# (this method memoizes the model in the repository)

# Method 3: Using create_model_from_spec function
model = create_model_from_spec(model="OpenAI/gpt-4o")
# (this method memoizes the model in the repository)

# Method 4: Using dictionary unpacking
spec = {'model': "OpenAI/gpt-4o", 'temperature': 0.7}
model = create_model_from_spec(**spec)
# (this method memoizes the model in the repository)
```

Behaviour:
    Raises exception from Langchain and from itself

Note:
    Support for new model sources should be added here by extending
    the match ... case statement in _create_model_instance.

    All methods support configurable parameters like temperature,
    max_tokens, max_retries, and timeout through
    LanguageModelSettings.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from ..lazy_dict import LazyLoadingDict

from lmm.config.config import (
    LanguageModelSettings,
    EmbeddingSettings,
    ModelSource,
    SparseModel,
)
from lmm.markdown.parse_yaml import MetadataPrimitiveWithList


# Factory function to create model. Here, the model definition is
# provided by the settings object that can be read from config.toml,
# LanguageModelSettings
def _create_model_instance(
    model: LanguageModelSettings,
) -> BaseChatModel:
    """
    Factory function to create Langchain models while checking
    permissible sources.
    """
    model_source: ModelSource = model.get_model_source()
    model_name: str = model.get_model_name()
    match model_source:
        case "Anthropic":
            try:
                from langchain_anthropic.chat_models import (
                    ChatAnthropic,
                )
            except ImportError as e:
                raise ImportError(
                    "Anthropic models require the "
                    "'langchain-anthropic' package. "
                    "Install it with: pip install langchain-anthropic"
                ) from e

            # Build kwargs dict to handle optional parameters
            kwargs: dict[str, Any] = {
                "model_name": model_name,
                "temperature": model.temperature,
                "max_tokens_to_sample": model.max_tokens or 1024,
                "timeout": model.timeout,
                "max_retries": model.max_retries,
                "stop": None,
            }

            # Add provider-specific parameters
            kwargs.update(model.provider_params)

            return ChatAnthropic(**kwargs)

        case "Gemini":
            try:
                from langchain_google_genai import (
                    ChatGoogleGenerativeAI,
                )
            except ImportError as e:
                raise ImportError(
                    "Gemini models require the "
                    "'langchain-google-genai' package. "
                    "Install it with: pip install "
                    "langchain-google-genai"
                ) from e

            # Build kwargs dict to handle optional parameters
            kwargs = {
                "model": model_name,
                "temperature": model.temperature,
            }
            if model.max_tokens is not None:
                kwargs["max_output_tokens"] = model.max_tokens
            if model.timeout is not None:
                kwargs["request_timeout"] = model.timeout

            # Add provider-specific parameters
            kwargs.update(model.provider_params)

            return ChatGoogleGenerativeAI(**kwargs)

        case "Mistral":
            try:
                from langchain_mistralai.chat_models import (
                    ChatMistralAI,
                )
            except ImportError as e:
                raise ImportError(
                    "Mistral models require the 'langchain-mistralai'"
                    " package. Install it with: pip install "
                    "langchain-mistralai"
                ) from e

            # Build kwargs dict to handle optional parameters
            kwargs = {
                "model_name": model_name,
                "temperature": model.temperature,
                "max_retries": model.max_retries,
            }
            if model.max_tokens is not None:
                kwargs["max_tokens"] = model.max_tokens
            if model.timeout is not None:
                kwargs["timeout"] = int(model.timeout)

            # Add provider-specific parameters
            kwargs.update(model.provider_params)

            return ChatMistralAI(**kwargs)

        case "OpenAI":
            try:
                from langchain_openai.chat_models import ChatOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI models require the 'langchain-openai'"
                    " package. Install it with: pip install "
                    "langchain-openai"
                ) from e

            # Build kwargs dict to handle optional parameters
            kwargs = {
                "model": model_name,
                "temperature": model.temperature,
                "max_retries": model.max_retries,
                "use_responses_api": False,
            }
            if model.max_tokens is not None:
                kwargs["max_tokens"] = model.max_tokens
            if model.timeout is not None:
                kwargs["timeout"] = model.timeout

            # Add provider-specific parameters
            kwargs.update(model.provider_params)

            return ChatOpenAI(**kwargs)

        case "Debug":
            from ..message_iterator import (
                yield_message,
                yield_constant_message,
            )

            try:
                from langchain_core.language_models.fake_chat_models import (
                    GenericFakeChatModel,
                )
            except ImportError as e:
                raise ImportError(
                    "Could not import GenericFakeChatModel from langchain_core"
                ) from e
            if (
                model.provider_params
                    and "message" in model.provider_params.keys()
            ):
                return GenericFakeChatModel(
                    name="Langchain fake messages",
                    messages=yield_constant_message(
                        str(model.provider_params["message"])
                    ),
                )
            else:
                return GenericFakeChatModel(
                    name="Langchain fake chat",
                    messages=yield_message(),
                )

        case _:
            raise ValueError(
                f"Unreachable code reached: invalid source {model_source}"
            )


# The specification of the embedding is read from config.toml too
def _create_embedding_instance(
    model: EmbeddingSettings,
) -> Embeddings:
    """
    Factory function to create Langchain models while checking
    permissible sources.
    """
    model_source: str = model.get_model_source()
    model_name: str = model.get_model_name()
    match model_source:
        case "Gemini":
            try:
                from langchain_google_genai import (
                    GoogleGenerativeAIEmbeddings,
                )
            except ImportError as e:
                raise ImportError(
                    "Gemini models require the "
                    "'langchain-google-genai' package. "
                    "Install it with: pip install langchain-google-genai"
                ) from e

            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                task_type="retrieval_document",
            )

        case "Mistral":
            try:
                from langchain_mistralai import MistralAIEmbeddings
            except ImportError as e:
                raise ImportError(
                    "Mistral models require the "
                    "'langchain-mistralai' package. "
                    "Install it with: pip install langchain-mistralai"
                ) from e

            return MistralAIEmbeddings(
                model=model_name,
            )

        case "OpenAI":
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError as e:
                raise ImportError(
                    "OpenAI models require the 'langchain-openai'"
                    " package. Install it with: pip install "
                    "langchain-openai"
                ) from e

            return OpenAIEmbeddings(model=model_name)

        case "SentenceTransformers":
            from langchain_huggingface import HuggingFaceEmbeddings

            source_name = "sentence-transformers"
            full_model_name = f"{source_name}/{model_name}"
            # We rely on defaults:
            # - device: automatically selects "cuda" if available, else "cpu"
            # - local_files_only: defaults to False (will download if missing)
            
            encode_kwargs = {"normalize_embeddings": True}
            
            return HuggingFaceEmbeddings(
                model_name=full_model_name,
                encode_kwargs=encode_kwargs,
            )


# Public interface----------------------------------------------
langchain_models: LazyLoadingDict[LanguageModelSettings, BaseChatModel] = \
    LazyLoadingDict(_create_model_instance)
langchain_embeddings: LazyLoadingDict[EmbeddingSettings, Embeddings] = \
    LazyLoadingDict(_create_embedding_instance)


def create_model_from_spec(
    model: str,
    *,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    max_retries: int = 2,
    timeout: float | None = None,
    provider_params: dict[str, MetadataPrimitiveWithList] = {},
) -> BaseChatModel:
    """
    Create langchain model from specifications.

    Args:
        model: the model in the form source/model, such as
            'OpenAI/gpt-4o'

    Returns:
        a Langchain model object.

    Raises ValuationError, TypeError, ValidationError

    Example:
        ```python
        spec = {'model': "OpenAI/gpt-4o-mini"}
        model = create_model_from_spec(**spec)
        ```
    """

    spec = LanguageModelSettings(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        timeout=timeout,
        provider_params=provider_params,
    )
    return langchain_models[spec]


def create_model_from_settings(
    settings: LanguageModelSettings,
) -> BaseChatModel:
    """
    Create langchain model from a LanguageModelSettings object.
    Raises a ValueError if the source argument is not supported.

    Args:
        settings: a LanguageModelSettings object containing model
            configuration.

    Returns:
        a Langchain model object.

    Raises ValuationError, TypeError, ValidationError

    Example:
        ```python
        # Create settings explicitly.
        config = LanguageModelSettings(
            model="OpenAI/gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            max_retries=3,
        )
        model = create_model_from_settings(config)
        response = model.invoke("Why is the sky blue?")

        # Load settings from config.toml.
        settings = Settings()
        system_prompt = "You are a helpful assistant."
        model = create_model_from_settings(
            settings.minor,
        )
        response = model.invoke("Why is the grass green?")
        ```
    """
    return langchain_models[settings]


def create_embedding_model_from_spec(
    dense_model: str, *, sparse_model: SparseModel = "Qdrant/bm25"
) -> Embeddings:
    """
    Create langchain embedding model from source_name and
    model_name. Raises a ValueError if the source_name argument
    is not supported.

    Args:
        dense_model: the model specification in the form source/model,
            such as 'OpenAI/text-embedding-3-small'

    Returns:
        a Langchain embeddings model object.

    Raises ValuationError, TypeError, ValidationError

    Example:
        ```python
        spec = {'dense_model': "OpenAI/text-embedding-3-small"}
        model = create_embedding_model_from_spec(**spec)
        ```
    """
    spec = EmbeddingSettings(
        dense_model=dense_model,
        sparse_model=sparse_model,
    )
    return langchain_embeddings[spec]


def create_embedding_model_from_settings(
    settings: EmbeddingSettings,
) -> Embeddings:
    """
    Create langchain embedding model from an EmbeddingSettings
    object. Raises a ValueError if the source argument
    is not supported.

    Args:
        settings: an EmbeddingSettings object containing model
            configuration.

    Returns:
        a Langchain embeddings model object.

    Raises ValuationError, TypeError, ValidationError

    Example:
        ```python
        settings = EmbeddingSettings(
            dense_model="OpenAI/text-embedding-3-small"
        )
        model = create_embedding_model_from_settings(settings)
        ```
    """
    return langchain_embeddings[settings]
