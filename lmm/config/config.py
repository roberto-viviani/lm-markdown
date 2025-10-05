"""
Read and write configuration file.

This file also contains the definitions of the models supported in the
package.
"""

from pathlib import Path
from typing import Any, Literal, Self
from lmm.markdown.parse_yaml import MetadataPrimitive

from pydantic import (
    Field,
    field_validator,
    model_validator,
    BaseModel,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# Define supported models. These models must also be defined
# in the model_selection.py file of the language framework
ModelSource = Literal[
    'OpenAI', 'Anthropic', 'Mistral', 'Gemini', 'Debug'
]
EmbeddingSource = Literal[
    'OpenAI', 'Mistral', 'Gemini', 'SentenceTransformers'
]
SparseModel = Literal['prithivida/Splade_PP_en_v1', 'Qdrant/bm25']

# Constants for better maintainability
DEFAULT_CONFIG_FILE = "config.toml"


class LanguageModelSettings(BaseModel):
    """
    Specification of language sources and models.

    Attributes:
        model: model specificaion
        temperature: float between 0.0 and 2.0
        max_tokens: max number of generated tokens
        max_retries: max number retries attempts
        timmeout: timeout when waiting for response
        provider_params: provider-specific parameters
    """

    # Required
    model: str = Field(
        description="Model specification in the form "
        + "'model_provider/model' (e.g., 'OpenAI/gpt-4o')"
    )

    # Common configurable parameters
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Controls randomness in model responses (0.0-2.0)",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens to generate",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum number of retry attempts",
    )
    timeout: float | None = Field(
        default=None, gt=0, description="Request timeout in seconds"
    )

    # Provider-specific parameters
    provider_params: dict[str, MetadataPrimitive] = Field(
        default_factory=dict,
        description="Provider-specific parameters (e.g., frequency_penalty for OpenAI)",
    )

    model_config = SettingsConfigDict(frozen=True, extra='forbid')

    def __hash__(self) -> int:
        """Make the object hashable by converting provider_params to a sorted tuple."""
        # Convert provider_params dict to a sorted tuple of items for hashing
        provider_params_tuple = tuple(
            sorted(self.provider_params.items())
        )
        return hash(
            (
                self.model,
                self.temperature,
                self.max_tokens,
                self.max_retries,
                self.timeout,
                provider_params_tuple,
            )
        )

    def get_model_source(self) -> ModelSource:
        return self.model.split('/')[0]  # type: ignore

    def get_model_name(self) -> str:
        return self.model.split('/')[1]

    # A method to create a new instance with some fields modified
    def from_instance(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
        provider_params: dict[str, MetadataPrimitive] | None = None,
    ) -> 'LanguageModelSettings':
        return LanguageModelSettings(
            model=model if model is not None else self.model,
            temperature=(
                temperature
                if temperature is not None
                else self.temperature
            ),
            max_tokens=(
                max_tokens
                if max_tokens is not None
                else self.max_tokens
            ),
            max_retries=(
                max_retries
                if max_retries is not None
                else self.max_retries
            ),
            timeout=timeout if timeout is not None else self.timeout,
            provider_params=(
                provider_params
                if provider_params is not None
                else self.provider_params
            ),
        )

    @field_validator('model', mode='after')
    @classmethod
    def validate_model_spec(cls, spec: str) -> str:
        cleaned_spec = spec.strip()
        if not (bool(cleaned_spec)):
            raise ValueError("Model specification is empty")
        if '\n' in cleaned_spec or '\r' in cleaned_spec:
            raise ValueError(
                "Model specification cannot contain newlines or carriage"
                + " returns."
            )
        tokens = cleaned_spec.split('/')
        if len(tokens) != 2:
            raise ValueError(
                "Model specification must contain the model provider and "
                + "the model name separated by a single '/'.",
            )
        model_spec = tokens[0].strip()
        if model_spec not in ModelSource.__args__:
            raise ValueError(
                f"Invalid model provider: '{model_spec}'. "
                + f"Must be one of {ModelSource.__args__}."
            )
        return model_spec + '/' + tokens[1].strip()

    @model_validator(mode='after')
    def validate_provider_params(self) -> Self:
        """Validate provider-specific parameters based on the source."""
        params = self.provider_params

        # Define allowed parameters per provider (keeping it simple)
        ALLOWED_PARAMS = {
            'OpenAI': {
                'frequency_penalty',
                'presence_penalty',
                'top_p',
                'seed',
                'logprobs',
                'top_logprobs',
            },
            'Anthropic': {'top_p', 'top_k', 'stop_sequences'},
            'Mistral': {'top_p', 'random_seed', 'safe_mode'},
            'Gemini': {'top_p', 'top_k', 'candidate_count'},
        }

        # Get source from the current validation context
        source: ModelSource = self.get_model_source()
        if source and source in ALLOWED_PARAMS:
            allowed = ALLOWED_PARAMS[source]
            invalid_params = set(params.keys()) - allowed

            if invalid_params:
                raise ValueError(
                    f"Invalid provider_params for {source}: {invalid_params}. Allowed: {allowed}"
                )

        return self


class EmbeddingSettings(BaseSettings):
    """
    Specification of embeddings object.

    Attributes:
        dense_model: embedding model specification
        sparse_model: sparse embeddings
    """

    dense_model: str = Field(
        description="Model specification in the form "
        + "'model_provider/model' (e.g., 'OpenAI/text-embedding-3-small')"
    )
    sparse_model: SparseModel = Field(
        default="Qdrant/bm25",  # multilingual
        description="Sparse embedding model for hybrid search",
    )

    model_config = SettingsConfigDict(frozen=True, extra='forbid')

    def get_model_source(self) -> EmbeddingSource:
        return self.dense_model.split('/')[0]  # type: ignore

    def get_model_name(self) -> str:
        return self.dense_model.split('/')[1]

    def get_sparse_model_name(self) -> str:
        return str(self.sparse_model)

    @field_validator('dense_model', mode='after')
    @classmethod
    def validate_model_spec(cls, spec: str) -> str:
        cleaned_spec = spec.strip()
        if not (bool(cleaned_spec)):
            raise ValueError("Model specification is empty")
        if '\n' in cleaned_spec or '\r' in cleaned_spec:
            raise ValueError(
                "Model specification cannot contain newlines or carriage"
                + " returns."
            )
        tokens = cleaned_spec.split('/')
        if len(tokens) != 2:
            raise ValueError(
                "Model specification must contain the model provider and "
                + "the model name separated by a single '/'."
            )
        model_spec = tokens[0].strip()
        if model_spec not in EmbeddingSource.__args__:
            raise ValueError(
                f"Invalid model provider: '{model_spec}'. "
                + f"Must be one of {EmbeddingSource.__args__}."
            )
        return model_spec + '/' + tokens[1].strip()


class Settings(BaseSettings):
    """
    A pydantic settings object containing the fields with the
    configuration information.

    Settings are saved and read from the configuration file in TOML
    format.

    Attributes:
        server: Server configuration settings
        embeddings: Embedding model configuration
        major: Primary language model for complex tasks
        minor: Secondary language model for simple tasks
        aux: Auxiliary language model for specialized tasks

    Note:
        At present, the Settings object only reads from config.toml in
        the project folder. This path and name can be customized via
        the model_config.
    """

    # Language model embeddings
    embeddings: EmbeddingSettings = Field(
        default_factory=lambda: EmbeddingSettings(
            dense_model="OpenAI/text-embedding-3-small"
        ),
        description="Embedding model configuration",
    )

    # Language models with better naming and validation
    major: LanguageModelSettings = Field(
        default_factory=lambda: LanguageModelSettings(
            model="OpenAI/gpt-4.1-mini",
        ),
        description="Primary language model for complex reasoning tasks",
    )
    minor: LanguageModelSettings = Field(
        default_factory=lambda: LanguageModelSettings(
            model="OpenAI/gpt-4.1-nano",
        ),
        description="Secondary language model for simple tasks",
    )
    aux: LanguageModelSettings = Field(
        default_factory=lambda: LanguageModelSettings(
            model="Mistral/mistral-small-latest", temperature=0.7
        ),
        description="Auxiliary language model for specialized tasks",
    )

    model_config = SettingsConfigDict(
        toml_file=DEFAULT_CONFIG_FILE,
        env_prefix="LMM_",  # Uppercase for environment variables
        frozen=True,
        validate_assignment=True,
        extra='allow',  # Do not prevent unexpected fields
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize the order of settings sources."""
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
        )

    def __string__(self) -> str:
        return serialize_settings(self)


def serialize_settings(sets: BaseSettings) -> str:
    """Transform the settings into a string in TOML format.

    Args:
        sets: The settings object to serialize

    Returns:
        TOML formatted string representation of settings

    Raises:
        ImportError: If tomlkit is not available
        ValueError: If settings cannot be serialized
    """
    try:
        import tomlkit
    except ImportError as e:
        raise ImportError(
            "tomlkit is required for TOML serialization"
        ) from e

    # try:
    doc = tomlkit.document()
    doc.add(tomlkit.comment("Configuration file"))
    doc.add(tomlkit.nl())

    data: dict[str, Any] = sets.model_dump()
    for key, value in data.items():
        if isinstance(value, dict):
            # Handle nested dictionaries (from BaseSettings objects)
            tbl = tomlkit.table()
            for kkey, vvalue in value.items():  # type: ignore
                # Skip None values as they can't be serialized to TOML
                if vvalue is not None:
                    tbl[kkey] = vvalue
            doc[key] = tbl
        else:
            # Skip None values at top level too
            if value is not None:
                doc[key] = value

    return str(tomlkit.dumps(doc))  # type: ignore


def export_settings(
    settings: BaseSettings, file_path: str | Path | None = None
) -> None:
    """Save settings to file in TOML format.

    Args:
        settings: A settings object to save
        file_path: The settings file path (defaults to config.toml)

    Raises:
        ImportError: If tomlkit is not available
        OSError: If file cannot be written
        ValueError: If settings cannot be serialized
    """
    if file_path is None:
        file_path = DEFAULT_CONFIG_FILE

    file_path = Path(file_path)

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        f.write(serialize_settings(settings))


def create_default_config_file(
    file_path: str | Path | None = None,
) -> None:
    """Create a default settings file.

    Args:
        file_path: Target file path (defaults to config.toml)

    Raises:
        ImportError: If tomlkit is not available
        OSError: If file cannot be written
        ValueError: If settings cannot be serialized

    Example:
        ```python
        # Creates config.toml in base folder with default values
        create_default_settings_file()

        # Creates custom config file
        create_default_settings_file(file_path="custom_config.toml")
        ```
    """
    if file_path is None:
        file_path = DEFAULT_CONFIG_FILE

    file_path = Path(file_path)

    if file_path.exists():
        # otherwise, it will be read in
        file_path.unlink()

    settings = Settings()

    export_settings(settings, file_path)


def print_settings(settings: BaseSettings) -> None:
    """Print settings in TOML format to stdout.

    Args:
        settings: The settings object to print

    Raises:
        ImportError: If tomlkit is not available
        ValueError: If settings cannot be serialized
    """
    print(serialize_settings(settings))


def load_settings(file_path: str | Path | None = None) -> Settings:
    """Load settings from TOML file.

    Args:
        file_path: Path to settings file (defaults to config.toml)

    Returns:
        Loaded settings object

    Raises:
        FileNotFoundError: If settings file doesn't exist
        ValueError: If settings file is invalid
    """
    if file_path is None:
        file_path = DEFAULT_CONFIG_FILE

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Settings file not found: {file_path}"
        )

    try:
        # Create a temporary settings class with the specified file
        class TempSettings(Settings):
            model_config = SettingsConfigDict(
                toml_file=str(file_path),
                env_prefix="LMM_",
                frozen=True,
                validate_assignment=True,
                extra='allow',
            )

        return TempSettings()
    except Exception as e:
        raise ValueError(
            f"Failed to load settings from {file_path}: {e}"
        ) from e


def format_pydantic_error_message(error_message: str) -> str:
    """Filter out verbose lines from pydantic error messages.

    Args:
        error_message: Raw pydantic error message

    Returns:
        Cleaned error message without verbose help text
    """
    lines = error_message.split('\n')
    filtered_lines = [
        line
        for line in lines
        if "For further information visit" not in line
    ]
    return '\n'.join(filtered_lines)


# Create a default config.toml file, if there is none.
if not Path(DEFAULT_CONFIG_FILE).exists():
    create_default_config_file()
