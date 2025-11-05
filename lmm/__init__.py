def _default_config() -> None:
    # Create a default config.toml file, if there is none.
    from pathlib import Path
    from config.config import (
        DEFAULT_CONFIG_FILE,
        create_default_config_file,
    )

    if not Path(DEFAULT_CONFIG_FILE).exists():
        create_default_config_file()


_default_config()
