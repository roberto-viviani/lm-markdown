"""
Read and write configuration file.

This file also contains the definitions of the models supported in the
package.
"""

from typing import Literal

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# define supported models. These models must also be defined
# in the model_selection.py file of the language framework
LMSource = Literal['OpenAI', 'Anthropic', 'Mistral', 'Gemini']
EmbSource = Literal[
    'OpenAI', 'Mistral', 'Gemini', 'SentenceTransformers'
]
SparseModel = Literal['prithivida/Splade_PP_en_v1', 'Qdrant/bm25']


class EmbeddingSettings(BaseSettings):
    """Specification of embeddings object."""

    source: EmbSource = "OpenAI"
    model: str = "text-embedding-3-small"
    sparse: str = "Qdrant/bm25"  # multilingual

    model_config = SettingsConfigDict(frozen=True)


class LanguageModelSettings(BaseSettings):
    """Specification of language sources and models."""

    source: LMSource = "OpenAI"
    model: str = ""

    model_config = SettingsConfigDict(frozen=True)


class ServerSettings(BaseSettings):
    """Server modality (Local/remote)"""

    mode: str = "local"
    port: int = 0


class Settings(BaseSettings):
    """
    A pydantic settings object containing the fields with the
    configuration information.

    Settings are saved and read from the configuration file in toml
    format.
    """

    server: ServerSettings = ServerSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    # language_models: LanguageModelSettings = LanguageModelSettings()
    major: LanguageModelSettings = LanguageModelSettings(
        source="OpenAI", model="gpt-4o-mini"
    )
    minor: LanguageModelSettings = LanguageModelSettings(
        source="OpenAI", model="gpt-4o-nano"
    )
    aux: LanguageModelSettings = LanguageModelSettings(
        source="Mistral", model="mistral-small-latest"
    )

    model_config = SettingsConfigDict(
        toml_file='config.toml', env_prefix="lmm_", frozen=True
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
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
        )


# utilities
def serialize_settings(sts: Settings) -> str:
    """Transform the settings into a string in toml format"""
    import tomlkit

    doc = tomlkit.document()
    doc.add(tomlkit.comment("Configuration file"))
    doc.add(tomlkit.nl())
    data: dict[str, str | BaseSettings] = sts.model_dump()
    for key, value in data.items():
        if isinstance(value, BaseSettings):
            subdata: dict[str, str] = value.model_dump()
            tbl = tomlkit.table()
            for kkey, vvalue in subdata.items():
                tbl[kkey] = vvalue
            doc[key] = tbl
        else:
            doc[key] = value

    return tomlkit.dumps(doc)  # type: ignore


def export_settings(sts: Settings, file: str = "config.toml") -> None:
    """Save settings to file in toml format.

    Args:
        sts: a settings object to save
        file: the settings toml file (deafults to config.toml)
    """
    with open(file, "w") as f:
        f.write(serialize_settings(sts))


def create_default_settings_file(
    sts: Settings = Settings(), file: str = "config.toml"
) -> None:
    """Create a default settings file.

    Example:
        ```python
        # creates config.toml in base folder with default values
        create_default_settings_file()
        ```
    """
    export_settings(sts, file)


def print_settings(sts: Settings) -> None:
    print(serialize_settings(sts))


def fmat_pydantic_errmsg(errmsg: str) -> str:
    """Filter out lines containing 'For further information visit'
    from error message."""

    lines = errmsg.split('\n')
    filtered_lines = [
        line
        for line in lines
        if "For further information visit" not in line
    ]
    return '\n'.join(filtered_lines)
