"""
Read and write configuration file.

This file also contains the definitions of the models supported in the
package.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# Define supported models. These models must also be defined
# in the model_selection.py file of the language framework
LMSource = Literal['OpenAI', 'Anthropic', 'Mistral', 'Gemini']
EmbSource = Literal[
    'OpenAI', 'Mistral', 'Gemini', 'SentenceTransformers'
]
SparseModel = Literal['prithivida/Splade_PP_en_v1', 'Qdrant/bm25']

# Constants for better maintainability
DEFAULT_CONFIG_FILE = "config.toml"
DEFAULT_PORT_RANGE = (
    1024,
    65535,
)  # Valid port range excluding system ports


class EmbeddingSettings(BaseSettings):
    """Specification of embeddings object."""

    source: EmbSource = Field(
        default="OpenAI", description="The embedding source provider"
    )
    name_model: str = Field(
        default="text-embedding-3-small",
        description="The specific embedding model name",
    )
    sparse: str = Field(
        default="Qdrant/bm25",  # multilingual
        description="Sparse embedding model for hybrid search",
    )

    model_config = SettingsConfigDict(frozen=True)

    @field_validator('name_model')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model name is not empty."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class LanguageModelSettings(BaseSettings):
    """Specification of language sources and models."""

    source: LMSource = Field(
        default="OpenAI", description="The language model provider"
    )
    name_model: str = Field(
        default="gpt-4-mini",
        description="The specific language model name",
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
    provider_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific parameters (e.g., frequency_penalty for OpenAI)",
    )

    model_config = SettingsConfigDict(frozen=True)

    def __hash__(self) -> int:
        """Make the object hashable by converting provider_params to a sorted tuple."""
        # Convert provider_params dict to a sorted tuple of items for hashing
        provider_params_tuple = tuple(
            sorted(self.provider_params.items())
        )
        return hash(
            (
                self.source,
                self.name_model,
                self.temperature,
                self.max_tokens,
                self.max_retries,
                self.timeout,
                provider_params_tuple,
            )
        )

    @field_validator('name_model')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model name is provided when needed."""
        # Allow empty for base class, but specific instances should validate
        return v.strip() if v.strip() else v

    @field_validator('provider_params')  # type: ignore
    @classmethod
    def validate_provider_params(
        cls, v: dict[str, Any], info: tuple[tuple[str, Any], ...]
    ) -> dict[str, Any]:
        """Validate provider-specific parameters based on the source."""
        if not v:  # Empty dict is always valid
            return v

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
        source = None
        if (
            hasattr(info, 'data')
            and info.data  # type: ignore
            and 'source' in info.data  # type: ignore
        ):
            source = info.data['source']  # type: ignore

        if source and source in ALLOWED_PARAMS:
            allowed = ALLOWED_PARAMS[source]
            invalid_params = set(v.keys()) - allowed

            if invalid_params:
                raise ValueError(
                    f"Invalid provider_params for {source}: {invalid_params}. Allowed: {allowed}"
                )

        return v


class ServerSettings(BaseSettings):
    """Server configuration settings."""

    mode: Literal["local", "remote"] = Field(
        default="local", description="Server deployment mode"
    )
    port: int = Field(
        default=0,
        ge=0,
        le=65535,
        description="Server port (0 for auto-assignment)",
    )
    host: str = Field(
        default="localhost", description="Server host address"
    )

    model_config = SettingsConfigDict(frozen=True)

    @field_validator('port')
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number is in acceptable range."""
        if v != 0 and not (
            DEFAULT_PORT_RANGE[0] <= v <= DEFAULT_PORT_RANGE[1]
        ):
            raise ValueError(
                f"Port must be 0 (auto-assign) or between "
                f"{DEFAULT_PORT_RANGE[0]} and {DEFAULT_PORT_RANGE[1]}"
            )
        return v


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

    server: ServerSettings = Field(
        default_factory=ServerSettings,
        description="Server configuration",
    )
    embeddings: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings,
        description="Embedding model configuration",
    )

    # Language models with better naming and validation
    major: LanguageModelSettings = Field(
        default_factory=lambda: LanguageModelSettings(
            source="OpenAI", name_model="gpt-4o-mini"
        ),
        description="Primary language model for complex reasoning tasks",
    )
    minor: LanguageModelSettings = Field(
        default_factory=lambda: LanguageModelSettings(
            source="OpenAI", name_model="gpt-4o-nano"
        ),
        description="Secondary language model for simple tasks",
    )
    aux: LanguageModelSettings = Field(
        default_factory=lambda: LanguageModelSettings(
            source="Mistral", name_model="mistral-small-latest"
        ),
        description="Auxiliary language model for specialized tasks",
    )

    model_config = SettingsConfigDict(
        toml_file=DEFAULT_CONFIG_FILE,
        env_prefix="LMM_",  # Uppercase for environment variables
        frozen=True,
        validate_assignment=True,
        extra='forbid',  # Prevent unexpected fields
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

    @field_validator('major', 'minor', 'aux')
    @classmethod
    def validate_language_models(
        cls, v: LanguageModelSettings
    ) -> LanguageModelSettings:
        """Ensure language models have valid names."""
        if not v.name_model.strip():
            raise ValueError("Language model name cannot be empty")
        return v


# Utility functions with improved error handling and type safety
def serialize_settings(settings: Settings) -> str:
    """Transform the settings into a string in TOML format.

    Args:
        settings: The settings object to serialize

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

    data: dict[str, Any] = settings.model_dump()
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
    # except Exception as e:
    #     raise ValueError(f"Failed to serialize settings: {e}") from e


def export_settings(
    settings: Settings, file_path: str | Path | None = None
) -> None:
    """Save settings to file in TOML format.

    Args:
        settings: A settings object to save
        file_path: The settings file path (defaults to config.toml)

    Raises:
        OSError: If file cannot be written
        ValueError: If settings cannot be serialized
    """
    if file_path is None:
        file_path = DEFAULT_CONFIG_FILE

    file_path = Path(file_path)

    # try:
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        f.write(serialize_settings(settings))
    # except OSError as e:
    #     raise OSError(
    #         f"Failed to write settings to {file_path}: {e}"
    #     ) from e


def create_default_settings_file(
    settings: Settings | None = None,
    file_path: str | Path | None = None,
) -> None:
    """Create a default settings file.

    Args:
        settings: Custom settings object (defaults to Settings())
        file_path: Target file path (defaults to config.toml)

    Example:
        ```python
        # Creates config.toml in base folder with default values
        create_default_settings_file()

        # Creates custom config file
        create_default_settings_file(file_path="custom_config.toml")
        ```
    """
    if settings is None:
        settings = Settings()

    export_settings(settings, file_path)


def print_settings(settings: Settings) -> None:
    """Print settings in TOML format to stdout.

    Args:
        settings: The settings object to print
    """
    print(serialize_settings(settings))


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
                extra='forbid',
            )

        return TempSettings()
    except Exception as e:
        raise ValueError(
            f"Failed to load settings from {file_path}: {e}"
        ) from e
