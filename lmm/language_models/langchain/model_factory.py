"""This module implements creation of Langchain models from
a dictionary specification.

Examples:

```python
from lmm.language_models.langchain import langchain_factory, LanguageModelSettings
from lmm.language_models.langchain import create_model_from_spec

spec = Settings('source': "OpenAI", 'name_model': "gpt-4o")
model = langchain_factory[spec]

# alterantive
model = create_model_from_settings(spec)

# alternative
spec = {'source': "OpenAI", 'name_model': "gpt-4o"}
model = create_model_from_spec(**spec)
```
"""

from langchain_core.language_models.base import (
    BaseLanguageModel as BaseLM,
)
from langchain_core.messages import BaseMessage as BaseMsg
from langchain_core.embeddings import Embeddings

from ..lazy_dict import LazyLoadingDict

from lmm.config import LanguageModelSettings, EmbeddingSettings


# Langchain model type specified here.
def _create_model_instance(
    model: LanguageModelSettings,
) -> BaseLM[BaseMsg]:
    """
    Factory function to create Langchain models while checking permissible
    sources.
    """
    model_name: str = str(model.name_model)
    match model.source:
        case 'Anthropic':
            from langchain_anthropic.chat_models import ChatAnthropic

            return ChatAnthropic(
                model_name=model_name,
                temperature=0.1,
                max_tokens_to_sample=1024,
                timeout=None,
                max_retries=2,
                stop=None,
            )

        case 'Gemini':
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=model.name_model)

        case 'Mistral':
            from langchain_mistralai.chat_models import ChatMistralAI

            return ChatMistralAI(
                model_name=model_name,
                temperature=0.7,
            )

        case 'OpenAI':
            from langchain_openai.chat_models import ChatOpenAI

            return ChatOpenAI(
                model=model_name,
                temperature=0.1,
                max_retries=2,
                use_responses_api=False,
            )


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
            model_name = (model_name,)
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            return HuggingFaceEmbeddings(
                model_name=str(source_name) + "/" + str(model_name),
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )


# Public interface----------------------------------------------
langchain_factory = LazyLoadingDict(_create_model_instance)
langchain_embeddings = LazyLoadingDict(_create_embedding_instance)


def create_model_from_spec(
    source_name: str, model_name: str
) -> BaseLM[BaseMsg]:
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
) -> BaseLM[BaseMsg]:
    """Create langchain model from a LanguageModelSettings object.
    Raises a ValueError if the source_name argument is not supported.

    Args:
        settings: a settings object.

    Returns:
        a Langchain model object.

    Example:
        ```python
        settings = LanguageModelSettings{'source': "OpenAI", 'name_model': "gpt-4o-mini"}
        model = create_model_from_settings(spec)
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
        model = create_model_from_spec(**spec)
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
    object. Raises a ValueError if the source_name argument
    is not supported.

    Args:
        settings: the settings object.

    Returns:
        a Langchain embeddings model object.

    Example:
        ```python
        settings = EmbeddingSettings('source': "OpenAI",
                'name_model': "text-embedding-3-small")
        model = create_model_from_spec(spec)
        ```
    """
    return langchain_embeddings[settings]
