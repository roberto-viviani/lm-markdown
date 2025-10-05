"""
Creates Langchain runnable objects or 'kernels'. These objects may be
used in Langchain chains.

The runnable plugs two library resources into the Langchain interface:

- a language model access, selected via the models module from the
    specification in a config.toml file
- a set of tools that specialize the function of the language model,
    selected from the tool library provided by the tools module.

The tools module contains a set of predefined tools, allowing one to
create the specialized runnable/'kernel' from the kernel name.

The runnables are callable objects via the `invoke` member function.
The Langchain syntax is used with invoke, for example by passing a
dictionary that contains the parameters for the prompt template.

Example of a runnable created from a predefined tool:
    ```python
    from lmm.language_models.langchain.runnables import create_runnable
    query_model = create_runnable("query")  # uses config.toml for model

    # a runnable that specifies the model directly
    questions_model = create_runnable("question_generator",
                                {'model': "OpenAI/gpt-4o"})

    # use Langchain syntax to call the kernel
    try:
        response = questions_model.invoke({
            'text': "Logistic regression is typically used when the "
                + "outcome variable is binary."
        })
    except Exception:
        print("Could not obtain response from model")
    ```

Example of a dynamically created chat kernel:
    ```python
        from lmm.language_models.tools import (
            tool_library,
            create_prompt,
        )

        # this creates a prompt tool and registers it in the tool library
        prompt_template = '''Provide the questions to which the text answers.
            TEXT:
            {text}
        '''
        create_prompt(prompt_template, name = "question_generator")

        # create a kernel from the major model in config.toml with
        # this prompt
        from lmm.config.config import Settings
        from lmm.language_models.langchain.runnables import create_runnable
        settings = Settings()
        model = create_runnable(
            "question_generator",
            settings.major,
            "You are a helpful teacher")

        # if no settings object given, defaults to settings.minor
        model_minor = create_runnable("question_generator")
    ```

Note:
    A Langchain language model may be used directly by
    create_model_from_spec in the models module.
"""

from pydantic import BaseModel, ConfigDict

from langchain_core.runnables.base import (
    RunnableSerializable,
)
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.language_models.chat_models import BaseChatModel
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
from ..tools import (
    ToolDefinition,
    KernelNames,
    tool_library,
)


# Defines the runnable
class RunnableDefinition(BaseModel):
    """Groups together all properties that define the runnable"""

    kernel_name: KernelNames
    settings: LanguageModelSettings
    system_prompt_override: str | None = None

    # required for hashability
    model_config = ConfigDict(frozen=True, extra='forbid')


# Defines the embedding model (we use the settings directly)
EmbeddingModel = EmbeddingSettings

# Exports the type of the Langchain object
RunnableType = RunnableSerializable[dict[str, str], str]


# The factory functions
def _create_runnable(
    model: RunnableDefinition,
) -> RunnableType:  # RunnableSerializable[dict[str, str], str]
    """Assembles a Langchain chain with a prompt from kernel_prompts
    and a language model as specified by a LanguageModelSettings."""

    # fetch the kernel definition from the library
    kernel_definition: ToolDefinition = tool_library[
        model.kernel_name
    ]
    system_prompt = (
        kernel_definition.system_prompt
        if model.system_prompt_override is None
        else model.system_prompt_override
    )
    human_prompt = kernel_definition.prompt

    # Langchain prompt
    prompt: ChatPromptTemplate
    if system_prompt is not None:
        prompt = ChatPromptTemplate.from_messages(  # type: ignore
            [
                SystemMessagePromptTemplate.from_template(
                    system_prompt
                ),
                HumanMessagePromptTemplate.from_template(
                    human_prompt
                ),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_template(human_prompt)

    # the base language model. We customize the language model for
    # debug purposes to avoid calling the model provider.
    language_model: BaseChatModel
    language_model_settings: LanguageModelSettings = model.settings
    if language_model_settings.get_model_source() == "Debug":
        match model.kernel_name:
            case 'summarizer':
                language_model_settings.provider_params['message'] = (
                    "This is a summary of the text."
                )
            case 'question_generator':
                language_model_settings.provider_params['message'] = (
                    "These are questions the text answers."
                )
            case 'check_content':
                language_model_settings.provider_params['message'] = (
                    "statistics"
                )
            case _:
                # generic fake chat model in all other cases
                pass

    language_model = create_model_from_settings(
        language_model_settings
    )

    # combine into a runnable
    kernel: RunnableType = prompt | language_model | StrOutputParser()  # type: ignore
    # .name is a member function of RunnableSerializable
    # inited to None, which we reinitialize here
    kernel.name = (
        f"{model.kernel_name}:"
        + f"{model.settings.get_model_source()}/"
        + f"{model.settings.get_model_name()}"
    )

    return kernel


def _create_embedding(
    settings: EmbeddingSettings,
) -> Embeddings:
    """Assembles a Langchain embedding object from specs"""
    return create_embedding_model_from_settings(settings)


# global project-wide repository of kernels
runnable_library = LazyLoadingDict(_create_runnable)
embeddings_library = LazyLoadingDict(_create_embedding)


# Provides the shunting of the model to the major, minor, or aux
# specification provided in config.toml
def create_runnable(
    kernel_name: KernelNames | str,
    user_settings: (
        dict[str, str] | LanguageModelSettings | None
    ) = None,
    system_prompt: str | None = None,
) -> RunnableType:  # RunnableSerializable[dict[str, str], str]
    """
    Creates a Langchain kernel (a 'runnable') by combining tools/prompts
    created under the kernel_name parameters and configurations from
    config.toml with optional override settings.

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
            looks up in the tools_library dictionary if there is
            a prompt with that kernel_name, and returns a kernel
            object for a chat with that prompt.
        user_settings: Optional settings to override the default
            configuration. Can be either:
            - dict[str, str]: Dictionary with 'model' key
            - LanguageModelSettings: Pydantic model instance
            - None: Use settings from config.toml or defaults
        system_prompt: System prompt used in messages with the
            language model.

    Returns:
        A KernelType object RunnableSerializable[dict[str, str], str],
            A Langchain runnable chain that combines a prompt template,
            language model, and string output parser. The chain accepts a
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
        kernel = create_runnable("query")
        ```

        Override with dictionary:
        ```python
        kernel = create_runnable("summarizer",
            {"model": "OpenAI/gpt-4o"})
        ```

        Override with settings object:
        ```python
        from lmm.config.config import LanguageModelSettings
        settings = LanguageModelSettings(
            model="Mistral/mistral-small-latest"
        )
        kernel = create_runnable("question_generator", settings)
        ```

        The kernel object may be used with Langchain `.invoke` syntax:

        ```python
        response = kernel.invoke(
            {'text': "Logistic regression is used when the outcome"
                + " variable is binary."}
        )
        ```
    """

    def _create_or_get(
        settings: LanguageModelSettings,
        kernel_name: KernelNames,
        system_prompt: str | None,
    ) -> RunnableType:
        model: RunnableDefinition = RunnableDefinition(
            kernel_name=kernel_name,
            settings=settings,
            system_prompt_override=system_prompt,
        )
        return runnable_library[model]

    match kernel_name:
        case 'query' | 'query_with_context':
            if bool(user_settings):  # disallow empty dictionary
                settings = Settings(major=user_settings)  # type: ignore
            else:
                settings = Settings()
            return _create_or_get(
                settings.major, kernel_name, system_prompt
            )
        case 'question_generator' | 'summarizer':
            if bool(user_settings):  # disallow empty dictionary
                settings = Settings(minor=user_settings)  # type: ignore
            else:
                settings = Settings()
            return _create_or_get(
                settings.minor, kernel_name, system_prompt
            )
        case 'check_content':
            if bool(user_settings):  # disallow empty dictionary
                settings = Settings(aux=user_settings)  # type: ignore
            else:
                settings = Settings()
            return _create_or_get(
                settings.aux, kernel_name, system_prompt
            )
        case str():
            # allows custom chat kernels to be constructed from custom
            # prompt templates
            if bool(user_settings):  # disallow empty dictionary
                settings = Settings(minor=user_settings)  # type: ignore
            else:
                settings = Settings()
            language_settings = settings.minor
            if kernel_name not in tool_library.keys():
                raise ValueError(
                    f"Invalid kernel name: {kernel_name}"
                )
            kernel_definition: ToolDefinition = tool_library[kernel_name]  # type: ignore
            sys_prompt = (
                kernel_definition.system_prompt
                if system_prompt is None
                else system_prompt
            )
            return create_kernel_from_objects(
                human_prompt=kernel_definition.prompt,
                system_prompt=sys_prompt,
                language_model=language_settings,
            )
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

    return embeddings_library[settings]


def create_kernel_from_objects(
    human_prompt: str,
    *,
    system_prompt: str | None = None,
    language_model: (
        BaseChatModel | LanguageModelSettings | None
    ) = None,
) -> RunnableType:
    """
    Creates a Langchain runnable from a prompt template and
    a language settings object.

    Args:
        human_prompt: prompt text
        system_prompt: system prompt text
        language_model: either a Langchain BaseChatModel, or
            a LanguageModelSettings object, or None (default). In
            this latter case the language.minor from the config
            file is used to create the model.

    Returns:
        a Langchain runnable, a type aliased as `RunnableType`.

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

    # Langchain prompt
    prompt: ChatPromptTemplate
    if system_prompt is not None:
        prompt = ChatPromptTemplate.from_messages(  # type: ignore
            [
                SystemMessagePromptTemplate.from_template(
                    system_prompt
                ),
                HumanMessagePromptTemplate.from_template(
                    human_prompt
                ),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_template(human_prompt)

    # combine into a runnable
    kernel: RunnableType = prompt | language_model | StrOutputParser()  # type: ignore
    # .name is a member function of RunnableSerializable
    # inited to None, which we initialize here
    kernel.name = name
    return kernel
