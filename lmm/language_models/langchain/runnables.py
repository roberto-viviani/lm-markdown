"""
Creates 'runnable' objects or 'kernels'. These objects may be
used to execute language model tasks.

The runnable plugs two library resources into the interface:

- a language model object, selected via the models module from the 
    specification in a config.toml file.
- a set of prompts that specialize the function of the language model,
    selected from the prompt library provided by the prompts module.

The runnables are callable objects via the `invoke` member function.

Example of a runnable created from a predefined tool:
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

    # use syntax to call the kernel after creating it
    try:
        response = questions_model.invoke({
            'text': "Logistic regression is typically used when the "
                + "outcome variable is binary."
        })
    except Exception:
        print("Could not obtain response from model")
    ```
"""

from pydantic import BaseModel, ConfigDict

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
    create_embedding_model_from_settings,
)

from ..lazy_dict import LazyLoadingDict
from ..prompts import (
    ToolDefinition,
    KernelNames,
    tool_library,
    _create_tool,  # type: ignore
)
from ..agent import Agent
from .adapter import LangChainChatModel
from ..base import BaseChatModel

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

    kernel_name: KernelNames
    settings: LanguageModelSettings
    system_prompt_override: str | None = None
    params: RunnableParameterType = frozenset()

    # required for hashability
    model_config = ConfigDict(frozen=True, extra='forbid')


# Defines the embedding model (we use the settings directly)
EmbeddingModel = EmbeddingSettings

# Exports the type of the object
RunnableType = Agent


# The factory functions
def _create_runnable(
    model: RunnableDefinition,
) -> RunnableType:
    """Assembles an Agent with a prompt from kernel_prompts
    and a language model as specified by a LanguageModelSettings."""

    # fetch the kernel definition from the library
    # If there are params, we need to call _create_tool directly with them
    if model.params:
        params_dict = _runnable_par_to_dict(model.params)
        kernel_definition: ToolDefinition = _create_tool(
            model.kernel_name, **params_dict
        )
    else:
        kernel_definition: ToolDefinition = tool_library[
            model.kernel_name
        ]
    system_prompt = (
        kernel_definition.system_prompt
        if model.system_prompt_override is None
        else model.system_prompt_override
    )
    human_prompt = kernel_definition.prompt

    # the base language model. We customize the language model for
    # debug purposes to avoid calling the model provider.
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
            case 'allowed_content_validator':
                param_dict = _runnable_par_to_dict(model.params)
                param_value = param_dict.pop(
                    'allowed_content', ['statistics']
                )
                if isinstance(param_value, list):
                    param_value = str(param_value[0])
                language_model_settings.provider_params['message'] = (
                    str(param_value)
                )
            case _:
                # generic fake chat model in all other cases
                pass

    # Create the adapter
    chat_model = LangChainChatModel(language_model_settings)

    # Create the Agent
    agent_name = (
        f"{model.kernel_name}:"
        + f"{model.settings.get_model_source()}/"
        + f"{model.settings.get_model_name()}"
    )
    agent = Agent(
        model=chat_model,
        prompt=human_prompt,
        system_prompt=system_prompt,
        name=agent_name,
    )

    return agent


def _create_embedding(
    settings: EmbeddingSettings,
) -> Embeddings:
    """Assembles a Langchain embedding object from specs"""
    return create_embedding_model_from_settings(settings)


# global project-wide repository of kernels
runnable_library: LazyLoadingDict[
    RunnableDefinition, Agent
] = LazyLoadingDict(_create_runnable)
embeddings_library: LazyLoadingDict[EmbeddingSettings, Embeddings] = (
    LazyLoadingDict(_create_embedding)
)


# Provides the shunting of the model to the major, minor, or aux
# specification provided in config.toml
def create_runnable(
    kernel_name: KernelNames | str,
    user_settings: (
        dict[str, str] | LanguageModelSettings | Settings | None
    ) = None,
    system_prompt: str | None = None,
    **kwargs: MetadataPrimitiveWithList,
) -> RunnableType:
    """
    Creates a kernel (an 'Agent') by combining tools/prompts
    created under the kernel_name parameters and configurations from
    config.toml with optional override settings.

    The function maps different kernel types to their appropriate
    language model settings categories.

    Args:
        kernel_name: The name of the kernel to create.
        user_settings: Optional settings to override the default
            configuration.
        system_prompt: System prompt used in messages with the
            language model.

    Returns:
        An Agent object.
    """

    def _create_or_get(
        settings: LanguageModelSettings,
        kernel_name: KernelNames,
        system_prompt: str | None,
        **kwargs: MetadataPrimitiveWithList,
    ) -> RunnableType:
        model: RunnableDefinition = RunnableDefinition(
            kernel_name=kernel_name,
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
        case None:
            settings = Settings()
        case _:
            raise ValueError(
                f"Invalid model definition: {user_settings}"
            )

    match kernel_name:
        case 'query' | 'query_with_context':
            return _create_or_get(
                settings.major, kernel_name, system_prompt
            )
        case 'question_generator' | 'summarizer':
            return _create_or_get(
                settings.minor, kernel_name, system_prompt
            )
        case 'allowed_content_validator' | 'context_validator':
            return _create_or_get(
                settings.aux,
                kernel_name,
                "You are a helpful assistant",
                **kwargs,
            )
        case str():
            # allows custom chat kernels to be constructed from custom
            # prompt templates
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
    settings: (
        dict[str, str] | EmbeddingSettings | Settings | None
    ) = None,
) -> Embeddings:
    """
    Creates a Langchain embeddings kernel from a configuration
    object.
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
    Creates an Agent from a prompt template and a language settings object.
    """
    if language_model is None:
        settings = Settings()
        chat_model = LangChainChatModel(settings.minor)
        name = f"Custom:{settings.minor.get_model_source()}/{settings.minor.get_model_name()}"
    elif isinstance(language_model, Settings):
        chat_model = LangChainChatModel(language_model.minor)
        name = f"Custom:{language_model.minor.get_model_source()}/{language_model.minor.get_model_name()}"
    elif isinstance(language_model, LanguageModelSettings):
        chat_model = LangChainChatModel(language_model)
        name = f"Custom:{language_model.get_model_source()}/{language_model.get_model_name()}"
    elif isinstance(language_model, BaseChatModel):  # type: ignore
        chat_model = language_model
        name = "Custom"
    else:
        # Fallback or error? The original code handled BaseChatModel.
        # We assume it's a BaseChatModel or compatible.
        # But wait, original code handled LangChain BaseChatModel.
        # We should probably wrap it if it's a LangChain model, but we don't want to import LangChain types here if we can avoid it.
        # For now, let's assume it's one of our types or we raise error.
        raise ValueError("Invalid language_model type")

    agent = Agent(
        model=chat_model,
        prompt=human_prompt,
        system_prompt=system_prompt,
        name=name,
    )
    return agent
