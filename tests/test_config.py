"""Test settings module"""

# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportUnusedExpression=false

import unittest
import os

from lmm.config.config import (
    Settings,
    serialize_settings,
    export_settings,
    LanguageModelSettings,
    load_settings,
    create_default_config_file,
)

# This is whatever config.toml is at present
base_settings: Settings = Settings()


class TestSettings(unittest.TestCase):
    def test_create_settings(self):
        sets: Settings = Settings()
        conf: str = serialize_settings(sets)
        self.assertTrue(bool(conf))

    def test_default_settings(self):
        sets: Settings = Settings()
        self.assertEqual(base_settings, sets)

    def test_set_settings_given(self):
        sets: Settings = Settings(
            **{'minor': {'model': "OpenAI/gpt-4o"}}
        )
        self.assertEqual(sets.minor.get_model_name(), "gpt-4o")
        # unmentioned setting still set
        self.assertEqual(sets.aux.get_model_source(), "Mistral")
        self.assertEqual(
            sets.aux.get_model_name(), "mistral-small-latest"
        )

    def test_set_settings_given2(self):
        sets: Settings = Settings(
            **{'minor': {'model': "OpenAI  /gpt-4o"}}
        )
        self.assertEqual(sets.minor.get_model_source(), "OpenAI")
        self.assertEqual(sets.minor.get_model_name(), "gpt-4o")

    def test_set_settings_given3(self):
        sets: Settings = Settings(
            minor=LanguageModelSettings(
                **{'model': "OpenAI/gpt-4o-xxx"}
            )
        )
        self.assertEqual(sets.minor.get_model_name(), "gpt-4o-xxx")
        # unmentioned setting still set
        self.assertEqual(sets.aux.get_model_source(), "Mistral")
        self.assertEqual(
            sets.aux.get_model_name(), "mistral-small-latest"
        )

    def test_set_settings_given4(self):
        sets: Settings = Settings(
            minor=LanguageModelSettings(
                **{'model': "OpenAI/gpt-4o-xxx", 'temperature': 0.7}
            )
        )
        self.assertEqual(sets.minor.get_model_name(), "gpt-4o-xxx")
        self.assertEqual(sets.minor.temperature, 0.7)

    def test_set_settings_incomplete(self):
        # need to specify name model
        with self.assertRaises(ValueError):
            sets: Settings = Settings(
                **{'minor': {'model': "OpenAI"}}
            )
            sets

    def test_set_settings_incomplete2(self):
        # need to specify name model
        with self.assertRaises(ValueError):
            sets: Settings = Settings(
                minor=LanguageModelSettings(model="Gemini")
            )
            sets

    def test_set_settings_incomplete3(self):
        # need to specify name model
        with self.assertRaises(ValueError):
            sets: Settings = Settings(minor=LanguageModelSettings()) # type: ignore (intentional)
            sets

    def test_set_settings_incomplete4(self):
        # need to specify name model
        with self.assertRaises(ValueError):
            sets: Settings = Settings(
                minor=LanguageModelSettings(**{})
            )
            sets

    def test_set_settings_invalid_dict(self):
        # need to specify valid fields
        with self.assertRaises(ValueError):
            sets: Settings = Settings(
                minor=LanguageModelSettings(
                    **{
                        'model': "OpenAI/gpt-4o",
                        'peppa': "this",
                    }
                )
            )
            sets

    def test_set_settings_invalid_dict2(self):
        # need to specify valid fields
        with self.assertRaises(ValueError):
            sets: Settings = Settings(
                **{
                    'minor': {
                        'model': "OpenAI/gpt-4o",
                        'peppa': "this",
                    }
                }
            )
            sets

    # now new field can be specified as it facilitates
    # customization
    # def test_set_settings_invalid_dict3(self):
    #     # need to specify valid fields
    #     with self.assertRaises(ValueError):
    #         sets: Settings = Settings(
    #             **{
    #                 'general': {
    #                     'model': "OpenAI/gpt-4o",
    #                 }
    #             }
    #         )
    #         sets

    def test_set_settings_empty(self):
        # cannot do this.
        with self.assertRaises(ValueError):
            sets: Settings = Settings(
                minor=LanguageModelSettings(**{})
            )
            sets

    def test_readwrite_settings(self):
        set = Settings(major={'model': "Gemini/gemini-latest"})
        export_settings(set, "config_test.toml")

        sets: Settings | None = load_settings(file_name="config_test.toml")
        if sets is None:
            raise ValueError("Could not load settings.")
        # expected result: name_model as in file
        self.assertEqual(sets.major.get_model_source(), "Gemini")
        self.assertEqual(set.major.get_model_name(), "gemini-latest")
        # unmentioned setting still set
        self.assertEqual(sets.aux.get_model_source(), "Mistral")
        self.assertEqual(
            sets.aux.get_model_name(), "mistral-small-latest"
        )
        os.unlink("config_test.toml")

    def test_set_settings_given_invalid(self):
        with self.assertRaises(ValueError):
            sets = Settings(minor={'OpenAI/4'})
            sets.minor  # for linter

    def test_set_settings_given_invalid_enum(self):
        # invalid literal: cohere not supported
        with self.assertRaises(ValueError):
            sets = Settings(
                **{
                    'major': {
                        'model': "cohere/latest",
                    }
                }
            )
            sets.major.model  # for linter

    def test_set_settings_invalid(self):
        sets: Settings = Settings()
        # model is frozen, otherwise would raise no error
        with self.assertRaises(ValueError):
            sets.minor.model = "Gemini/gemini-latest"
            sets.minor.model

    def test_default_config(self):
        create_default_config_file("temp_config.toml")
        default_sets: Settings | None = load_settings(file_name="temp_config.toml")
        if default_sets is None:
            raise ValueError("Could not load settings.")
        sets = Settings(
            **{'major': {'model': "Mistral/mistral-large-latest"}}
        )
        export_settings(sets, "temp_config.toml")
        create_default_config_file("temp_config.toml")
        sets: Settings | None = load_settings(file_name="temp_config.toml")
        if sets is None:
            raise ValueError("Could not load settings.")
        self.assertEqual(
            sets.major.get_model_source(),
            default_sets.major.get_model_source(),
        )
        os.unlink("temp_config.toml")


class TestGenericLoadSettings(unittest.TestCase):
    """Test the generic load_settings functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from pydantic_settings import (
            BaseSettings,
            PydanticBaseSettingsSource,
            SettingsConfigDict,
            TomlConfigSettingsSource,
        )
        from pydantic import Field

        # Define a custom settings class for testing
        class CustomSettings(BaseSettings):
            """A custom settings class for testing generics."""

            custom_field: str = Field(
                default="default_value",
                description="A custom field for testing",
            )
            custom_number: int = Field(
                default=42, description="A custom number field"
            )

            model_config = SettingsConfigDict(
                toml_file="custom_config.toml",
                extra='forbid',
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

        self.CustomSettings = CustomSettings
        self.test_file = "test_custom_config.toml"

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.unlink(self.test_file)

    def test_generic_load_with_custom_settings(self):
        """Test that load_settings works with a custom BaseSettings class."""
        # Create a custom settings instance
        custom = self.CustomSettings(
            custom_field="test_value", custom_number=100
        )

        # Export to a test file
        export_settings(custom, self.test_file)

        # Load it back using the generic function
        loaded = load_settings(
            file_name=self.test_file,
            settings_class=self.CustomSettings,
        )
        if loaded is None:
            raise ValueError("Could not load settings")

        # Verify it loaded correctly
        self.assertIsNotNone(loaded, "Settings should not be None")
        self.assertIsInstance(
            loaded,
            self.CustomSettings,
            "Should be CustomSettings instance",
        )
        self.assertEqual(
            loaded.custom_field, "test_value", "Field should match"
        )
        self.assertEqual(
            loaded.custom_number, 100, "Number should match"
        )

    def test_generic_type_inference(self):
        """Test that type inference works correctly with generic function."""
        # Create and export custom settings
        custom = self.CustomSettings(
            custom_field="inference_test", custom_number=999
        )
        export_settings(custom, self.test_file)

        # Load with explicit type
        loaded = load_settings(
            file_name=self.test_file,
            settings_class=self.CustomSettings,
        )
        if loaded is None:
            raise ValueError("Could not load settings")

        self.assertIsNotNone(loaded)
        # These attribute accesses should not raise AttributeError
        field_value: str = loaded.custom_field
        number_value: int = loaded.custom_number

        self.assertEqual(field_value, "inference_test")
        self.assertEqual(number_value, 999)

    def test_backward_compatibility_default_settings(self):
        """Test that default Settings class still works (backward compatibility)."""
        # Create and export default settings
        settings = Settings()
        test_file = "test_default_compat_config.toml"

        try:
            export_settings(settings, test_file)

            # Load without specifying settings_class (should default to Settings)
            loaded = load_settings(file_name=test_file)

            self.assertIsNotNone(
                loaded, "Settings should not be None"
            )
            self.assertIsInstance(
                loaded, Settings, "Should be Settings instance"
            )
            self.assertTrue(
                hasattr(loaded, 'major'),
                "Should have 'major' attribute",
            )
            self.assertTrue(
                hasattr(loaded, 'minor'),
                "Should have 'minor' attribute",
            )
            self.assertTrue(
                hasattr(loaded, 'aux'), "Should have 'aux' attribute"
            )

        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)


if __name__ == "__main__":
    unittest.main()
