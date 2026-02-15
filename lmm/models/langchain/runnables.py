"""
Creates Langchain 'runnable' objects or 'chain' members. These objects
may be combined to form Langchain chains, or used by themselves using
the .invoke/.ainvoke member functions. The created objects are made
available through the global variables `runnable_library` and 
`embeddings_library`, which memoize the objects.

Each object plugs two global resources into the Langchain interface:

- a language model object, itself wrapped by LangChain, selected via
    the models module from the specification in a config.toml file.
- a set of prompts that specialize the function of the language model,
    selected from the prompt library provided by the prompts module.

The prompts module contains a set of predefined prompts, allowing one
to create the specialized runnable/'chain' from the prompt name.

The runnables are callable objects via the `invoke` member function.
The LangChain syntax is used with invoke, for example by passing a
dictionary that contains the parameters for the prompt template.

Example of runnable creations:

    ```python
    from lmm.language_models.langchain.runnables import create_runnable
    try:
        query_model = create_runnable("query")  # uses config.toml
    except Exception ...

    # a runnable that specifies the model directly
    try:
        questions_model = create_runnable("question_generator",
                                {'model': "OpenAI/gpt-4o"})
    except Exception ...

    # use Langchain syntax to call the runnable after creating it
    try:
        response = questions_model.invoke({
            'text': "Logistic regression is typically used when the "
                + "outcome variable is binary."
        })
    except Exception:
        print("Could not obtain response from model")
    ```

Example of a dynamically created chat runnable:

    ```python
        from lmm.language_models.tools import (
            prompt_library,
            create_prompt,
        )

        # this creates a prompt and registers it in the prompts library
        prompt_template = '''Provide the questions to which the text answers.
            TEXT:
            {text}
        '''
        create_prompt(prompt_template, name = "question_generator")

        # create a runnable from the major model in config.toml with
        # this prompt
        from lmm.config.config import Settings
        from lmm.language_models.langchain.runnables import create_runnable
        settings = Settings()
        try:
            model = create_runnable(
                "question_generator",
                settings.major,
                "You are a helpful teacher")
        except Exception ...

        # if no settings object given, defaults to settings.minor
        try:
            model_minor = create_runnable("question_generator")
        except Exception ...
    ```

Expected behaviour:
    This module raises exceptions from Langhchain and itself.

Note:
    A Langchain language model may be used directly after obtaining it
        from create_model_from_spec in the models module.
"""
# review g + a

from pydantic import BaseModel, ConfigDict

from langchain_core.runnables.base import (
    RunnableSerializable,
)
from langchain_core.prompts import (
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
from lmm.markdown.parse_yaml import (
    MetadataPrimitive,
    MetadataPrimitiveWithList,
)
from .models import (
    create_model_from_settings,
    create_embedding_model_from_settings,
)

from ..lazy_dict import LazyLoadingDict
from ..prompts import (
    PromptDefinition,
    PromptNames,
    prompt_library,
    create_prompt_definition,
)

RunnableParameterValue = (
    MetadataPrimitive | tuple[MetadataPrimitive, ...]
)
RunnableParameterType = frozenset[tuple[str, RunnableParameterValue]]


def _runnable_par_to_dict(
    pars: RunnableParameterType,
) -> dict[str, MetadataPrimitiveWithList]:
    """Converts frozenset pars into dictionary"""
    temp: dict[str, RunnableParameterValue] = dict(pars)
    result: dict[str, MetadataPrimitiveWithList] = {}
    for item in temp.items():
        if isinstance(item[1], tuple):
            result[item[0]] = list(item[1])
        else:
            result[item[0]] = item[1]
    return result


def _dict_to_runnable_par(
    pars: dict[str, MetadataPrimitiveWithList],
) -> RunnableParameterType:
    """Convert pars into frozenset with tuple elements instead of lists"""
    temp: dict[str, RunnableParameterValue] = {}
    for item in pars.items():
        value_: MetadataPrimitiveWithList = item[1]
        if isinstance(value_, list):
            temp[item[0]] = tuple(value_)
        else:
            temp[item[0]] = value_
    return frozenset(temp.items())


# Defines the runnable
class RunnableDefinition(BaseModel):
    """Groups together all properties that define the runnable"""

    runnable_name: PromptNames | str
    settings: LanguageModelSettings
    system_prompt_override: str | None = None
    params: RunnableParameterType = frozenset()

    # required for hashability
    model_config = ConfigDict(frozen=True, extra='forbid')


# Defines the embedding model (we use the settings directly)
EmbeddingModel = EmbeddingSettings

# Exports the type of the Langchain object
RunnableType = RunnableSerializable[dict[str, str], str]

# Debug mode messages for fake model responses
_DEBUG_MESSAGES: dict[str, str] = {
    'summarizer': "This is a summary of the text.",
    'question_generator': "These are questions the text answers.",
}


# The factory functions
def _create_runnable(
    model: RunnableDefinition,
) -> RunnableType:  # RunnableSerializable[dict[str, str], str]
    """Assembles a Langchain runnable with a prompt from kernel_prompts
    and a language model as specified by a LanguageModelSettings."""

    # fetch the runnable definition from the library
    # If there are params, we need to call create_prompt_definition directly with them
    prompt_definition: PromptDefinition
    if model.params:
        params_dict = _runnable_par_to_dict(model.params)
        prompt_definition = create_prompt_definition(
            model.runnable_name, **params_dict  # type: ignore
        )
    else:
        prompt_definition = prompt_library[
            model.runnable_name # type: ignore
        ]
    system_prompt = (
        prompt_definition.system_prompt
        if model.system_prompt_override is None
        else model.system_prompt_override
    )
    human_prompt = prompt_definition.prompt

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

    # The language model is constructed here explicitly when
    # the provider is 'Debug' to avoid calling the model provider
    # and provide a string directly instead.
    language_model: BaseChatModel
    language_model_settings: LanguageModelSettings = model.settings
    if language_model_settings.get_model_source() == "Debug":
        # Use predefined debug messages or handle special cases
        if model.runnable_name in _DEBUG_MESSAGES:
            language_model_settings.provider_params['message'] = (
                _DEBUG_MESSAGES[model.runnable_name]
            )
        elif model.runnable_name == 'allowed_content_validator':
            param_dict = _runnable_par_to_dict(model.params)
            param_value = param_dict.pop(
                'allowed_content', ['statistics']
            )
            if not isinstance(param_value, list):
                raise TypeError(
                    f"allowed_content must be a list[str], got {type(param_value).__name__}"
                )
            language_model_settings.provider_params['message'] = (
                str(param_value[0])
            )
        # else: generic fake chat model (default behavior)

    language_model = create_model_from_settings(
        language_model_settings
    )

    # combine into a runnable
    runnable: RunnableType = prompt | language_model | StrOutputParser()  # type: ignore
    # .name is a member function of RunnableSerializable
    # inited to None, which we reinitialize here
    runnable.name = (
        f"{model.runnable_name}:"
        + f"{model.settings.get_model_source()}/"
        + f"{model.settings.get_model_name()}"
    )

    return runnable


def _create_embedding(
    settings: EmbeddingSettings,
) -> Embeddings:
    """Assembles a Langchain embedding object from specs"""
    return create_embedding_model_from_settings(settings)


# global project-wide repository of kernels
runnable_library: LazyLoadingDict[
    RunnableDefinition, RunnableSerializable[dict[str, str], str]
] = LazyLoadingDict(_create_runnable)
embeddings_library: LazyLoadingDict[EmbeddingSettings, Embeddings] = (
    LazyLoadingDict(_create_embedding)
)


# Provides the shunting of the model to the major, minor, or aux
# specification provided in config.toml
def create_runnable(
    runnable_name: PromptNames | str,
    user_settings: (
        dict[str, str] | LanguageModelSettings | Settings | None
    ) = None,
    system_prompt: str | None = None,
    **kwargs: MetadataPrimitiveWithList,
) -> RunnableType:  # RunnableSerializable[dict[str, str], str]
    """
    Creates a Langchain chain (a 'runnable') by combining tools/prompts
    created under the runnable_name parameters and configurations from
    config.toml with optional override settings.

    The function maps different chains to their appropriate
    language model settings categories. For example,
    - 'query', 'query_with_context' -> major model settings
    - 'question_generator', 'summarizer' -> minor model settings
    - 'allowed_content_validator', 'context_validator' ->
                                            aux model settings

    Settings Hierarchy (highest to lowest priority):
    1. user_settings parameter (if provided)
    2. config.toml file settings
    3. Default settings from Settings class

    Args:
        runnable_name: The name of the runnable to create. If one of
            the supported names defined in the PromptNames
            literal type, returns a cached runnable object. Otherwise,
            looks up in the tools_library dictionary if there is
            a prompt with that runnable_name, and returns a runnable
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
        ValueError: If runnable_name is not supported or if user_settings
            contains invalid model source names. No check is made at this
            stage that the model names are correct (as they frequently
            change); instead, failure occurs when the .invoke member function
            is called.
        ValidationError, TypeError: alternative errors raised in the same
            circumstances as above.
        ImportError: for not installed libraries.

    Examples:
        Create runnable with default settings read from configuration
        file:
        ```python
        try:
            runnable = create_runnable("query")
        except Exception ...
        ```

        Override with dictionary:
        ```python
        try:
            runnable = create_runnable("summarizer",
                {"model": "OpenAI/gpt-4o"})
        except Exception ...
        ```

        Override with settings object:
        ```python
        from lmm.config.config import LanguageModelSettings
        settings = LanguageModelSettings(
            model="Mistral/mistral-small-latest"
        )
        try:
            runnable = create_runnable("question_generator", settings)
        except Exception ...
        ```

        The runnable object may be used with Langchain `.invoke` syntax:

        ```python
        try:
            response = runnable.invoke(
                {'text': "Logistic regression is used when the outcome"
                    + " variable is binary."}
            )
        except Exception ...
        ```
    """

    def _create_or_get(
        settings: LanguageModelSettings,
        runnable_name: PromptNames,
        system_prompt: str | None,
        **kwargs: MetadataPrimitiveWithList,
    ) -> RunnableType:
        model: RunnableDefinition = RunnableDefinition(
            runnable_name=runnable_name,
            settings=settings,
            system_prompt_override=system_prompt,
            params=_dict_to_runnable_par(kwargs),
        )
        return runnable_library[model]

    settings: Settings
    match user_settings:
        case dict() if bool(user_settings):
            try:
                settings = Settings(
                    major=user_settings,  # type: ignore
                    minor=user_settings,  # type: ignore
                    aux=user_settings,  # type: ignore
                )
            except Exception as e:
                raise ValueError(f"Invalid model definition:\n{e}")
        case LanguageModelSettings():
            settings = Settings(
                major=user_settings,
                minor=user_settings,
                aux=user_settings,
            )
        case Settings():
            settings = user_settings
        case None | {}:
            settings = Settings()
        case _:
            raise ValueError(
                f"Invalid model definition: {user_settings}"
            )

    # Logic to retrieve or create prompt definition to check model tier
    prompt_definition: PromptDefinition
    params_dict = dict(kwargs)
    
    # We use create_prompt_definition if we have params (to handle validation)
    # or prompt_library if we don't (to handle custom prompts).
    if params_dict:
        try:
            prompt_definition = create_prompt_definition(runnable_name, **params_dict) # type: ignore
        except ValueError:
            # Fallback: if create_prompt_definition fails (e.g. custom prompt), 
            # check library. But strictly custom prompts shouldn't have params 
            # in current architecture unless handled by create_prompt_definition.
            # raising the error is appropriate if params were intended for a 
            # prompt that doesn't support them.
            raise
    else:
        try:
            prompt_definition = prompt_library[runnable_name] # type: ignore
        except Exception as e:
            raise ValueError(f"{runnable_name} is not a valid runnable name ({e})")

    match prompt_definition.model_tier:
        case 'major':
            settings_to_use = settings.major
        case 'minor':
            settings_to_use = settings.minor
        case 'aux':
            settings_to_use = settings.aux
        case _:
            settings_to_use = settings.minor

    return _create_or_get(
        settings_to_use, 
        runnable_name, # type: ignore
        system_prompt,
        **kwargs
    )


def create_embeddings(
    settings: (
        dict[str, str] | EmbeddingSettings | Settings | None
    ) = None,
) -> Embeddings:
    """
    Creates a Langchain embeddings runnable from a configuration
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
        requests.exceptions.ConnectionError: if not online

    Example:
    ```python
    from lmm.language_models.langchain.runnables import (
        create_embeddings,
    )

    try:
        encoder: Embeddings = create_embeddings()
        vector = encoder.embed_query("Why is the sky blue?")
        documents = ["The sky is blue due to its oxygen content"]
        vectors = encoder.embed_documents(documents)
    except Exception ...
    ```
    """
    if not bool(settings):  # includes empty dict
        sets = Settings()
        settings = sets.embeddings
    elif isinstance(settings, Settings):
        settings = settings.embeddings
    elif isinstance(settings, dict):
        # checked by pydantic model
        settings = EmbeddingSettings(**settings)  # type: ignore

    return embeddings_library[settings]


def create_kernel_from_objects(
    human_prompt: str,
    *,
    system_prompt: str | None = None,
    language_model: (
        BaseChatModel | LanguageModelSettings | Settings | None
    ) = None,
) -> RunnableType:
    """
    Creates a Langchain runnable from a prompt template and
    a language settings object. This name is not registered in the
    prompts library; it is available directly.

    Args:
        human_prompt: prompt text
        system_prompt: system prompt text
        language_model: either a Langchain BaseChatModel, or
            a LanguageModelSettings object, or None (default). In
            this latter case the language.minor from the config
            file is used to create the model.

    Returns:
        a Langchain runnable, a type aliased as `RunnableType`.

    Example:
    ```python
    human_prompt = '''
    Provide the questions to which the text answers.

    TEXT:
    {text}
    '''
    settings = Settings()
    try:
        model = create_kernel_from_objects(
            human_prompt=human_prompt,
            system_prompt="You are a helpful assistant",
            language_model=settings.aux,
        )
    except Exception ...

    # model use:
    try:
        response = model.invoke({'text': "Logistic regression is used"
            + " when the outcome variable is binary"})
    except Exception ...
    ```
    """
    if language_model is None:
        settings = Settings()
        language_model = create_model_from_settings(settings.minor)
        name = f"Custom:{settings.minor}"
    elif isinstance(language_model, Settings):
        name = f"Custom:{language_model.minor}"
        language_model = create_model_from_settings(
            language_model.minor
        )
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
    runnable: RunnableType = prompt | language_model | StrOutputParser()  # type: ignore
    # .name is a member function of RunnableSerializable
    # inited to None, which we re-initialize here
    runnable.name = name
    return runnable
