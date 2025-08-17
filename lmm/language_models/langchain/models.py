"""This module implements creation of Langchain models from
a dictionary specification.

Examples:

```python
from lmm.language_models.langchain.models import (
    create_model_from_spec,
    create_model_from_settings,
    langchain_factory,
)
from lmm.config import LanguageModelSettings

# Method 1: Using LanguageModelSettings object
settings = LanguageModelSettings(
    source="OpenAI",
    name_model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    max_retries=3
)
model = langchain_factory[settings]

# Method 2: Using create_model_from_settings function
model = create_model_from_settings(settings)

# Method 3: Using create_model_from_spec function
model = create_model_from_spec(source_name="OpenAI", model_name="gpt-4o")

# Method 4: Using dictionary unpacking
spec = {'source_name': "OpenAI", 'model_name': "gpt-4o"}
model = create_model_from_spec(**spec)
```

Note:
    Support for new model sources should be added here.
    All methods support configurable parameters like temperature, max_tokens,
    max_retries, and timeout through LanguageModelSettings.
"""

# pyright: reportArgumentType=false

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from ..lazy_dict import LazyLoadingDict

from lmm.config import LanguageModelSettings, EmbeddingSettings


# Langchain model type specified here.
def _create_model_instance(
    model: LanguageModelSettings,
) -> BaseChatModel:
    """
    Factory function to create Langchain models while checking permissible
    sources.
    """
    model_name: str = str(model.name_model)
    match model.source:
        case 'Anthropic':
            try:
                from langchain_anthropic.chat_models import (
                    ChatAnthropic,
                )
            except ImportError as e:
                raise ImportError(
                    "Anthropic models require the 'langchain-anthropic' package. "
                    "Install it with: pip install langchain-anthropic"
                ) from e

            # Build kwargs dict to handle optional parameters
            kwargs = {
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

        case 'Gemini':
            try:
                from langchain_google_genai import (
                    ChatGoogleGenerativeAI,
                )
            except ImportError as e:
                raise ImportError(
                    "Gemini models require the 'langchain-google-genai' package. "
                    "Install it with: pip install langchain-google-genai"
                ) from e

            # Build kwargs dict to handle optional parameters
            kwargs = {
                "model": model.name_model,
                "temperature": model.temperature,
            }
            if model.max_tokens is not None:
                kwargs["max_output_tokens"] = model.max_tokens
            if model.timeout is not None:
                kwargs["request_timeout"] = model.timeout

            # Add provider-specific parameters
            kwargs.update(model.provider_params)

            return ChatGoogleGenerativeAI(**kwargs)

        case 'Mistral':
            try:
                from langchain_mistralai.chat_models import (
                    ChatMistralAI,
                )
            except ImportError as e:
                raise ImportError(
                    "Mistral models require the 'langchain-mistralai' package. "
                    "Install it with: pip install langchain-mistralai"
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

        case 'OpenAI':
            try:
                from langchain_openai.chat_models import ChatOpenAI
            except ImportError as e:
                raise ImportError(
                    "OpenAI models require the 'langchain-openai' package. "
                    "Install it with: pip install langchain-openai"
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


# Langchain model type specified here.
def _create_embedding_instance(
    model: EmbeddingSettings,
) -> Embeddings:
    """
    Factory function to create Langchain models while checking permissible
    sources.
    """
    model_name = str(model.name_model)
    match model.source:
        case 'Gemini':
            from langchain_google_genai import (
                GoogleGenerativeAIEmbeddings,
            )

            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                task_type="retrieval_document",
            )

        case 'Mistral':
            from langchain_mistralai import MistralAIEmbeddings

            return MistralAIEmbeddings(
                model=model_name,
            )

        case 'OpenAI':
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model_name)

        case 'SentenceTransformers':
            from langchain_huggingface import HuggingFaceEmbeddings

            source_name = "sentence-transformers"
            # Fix: model_name should remain a string, not converted to tuple
            full_model_name = f"{source_name}/{model_name}"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            return HuggingFaceEmbeddings(
                model_name=full_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )


# Public interface----------------------------------------------
langchain_factory = LazyLoadingDict(_create_model_instance)
langchain_embeddings = LazyLoadingDict(_create_embedding_instance)


def create_model_from_spec(
    source_name: str, model_name: str
) -> BaseChatModel:
    """Create langchain model from source_name and model_name.
    Raises a ValueError if the source_name argument is not supported.

    Args:
        source_name: the model source, such as 'OpenAI'
        model_name: the name of the model, such as 'gpt-4o'

    Returns:
        a Langchain model object.

    Example:
        ```python
        spec = {'source_name': "OpenAI", 'model_name': "gpt-4o-mini"}
        model = create_model_from_spec(**spec)
        ```
    """
    spec = LanguageModelSettings(
        source=source_name,  # type: ignore
        name_model=model_name,
    )
    return langchain_factory[spec]


def create_model_from_settings(
    settings: LanguageModelSettings,
) -> BaseChatModel:
    """Create langchain model from a LanguageModelSettings object.
    Raises a ValueError if the source argument is not supported.

    Args:
        settings: a LanguageModelSettings object containing model configuration.

    Returns:
        a Langchain model object.

    Example:
        ```python
        settings = LanguageModelSettings(
            source="OpenAI",
            name_model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000
        )
        model = create_model_from_settings(settings)
        ```
    """
    return langchain_factory[settings]


def create_embedding_model_from_spec(
    source_name: str, model_name: str
) -> Embeddings:
    """Create langchain embedding model from source_name and
    model_name. Raises a ValueError if the source_name argument
    is not supported.

    Args:
        source_name: the model source, such as 'OpenAI'
        model_name: the name of the model, such as 'text-embedding-3-small'

    Returns:
        a Langchain embeddings model object.

    Example:
        ```python
        spec = {'source_name': "OpenAI",
                'model_name': "text-embedding-3-small"}
        model = create_embedding_model_from_spec(**spec)
        ```
    """
    spec = EmbeddingSettings(
        source=source_name,  # type: ignore
        name_model=model_name,
    )
    return langchain_embeddings[spec]


def create_embedding_model_from_settings(
    settings: EmbeddingSettings,
) -> Embeddings:
    """Create langchain embedding model from an EmbeddingSettings
    object. Raises a ValueError if the source argument
    is not supported.

    Args:
        settings: an EmbeddingSettings object containing model configuration.

    Returns:
        a Langchain embeddings model object.

    Example:
        ```python
        settings = EmbeddingSettings(
            source="OpenAI",
            name_model="text-embedding-3-small"
        )
        model = create_embedding_model_from_settings(settings)
        ```
    """
    return langchain_embeddings[settings]
