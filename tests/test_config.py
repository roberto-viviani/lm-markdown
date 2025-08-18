"""Test settings module"""

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
            sets: Settings = Settings(minor=LanguageModelSettings())
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

    def test_set_settings_invalid_dict3(self):
        # need to specify valid fields
        with self.assertRaises(ValueError):
            sets: Settings = Settings(
                **{
                    'general': {
                        'model': "OpenAI/gpt-4o",
                    }
                }
            )
            sets

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

        sets: Settings = load_settings("config_test.toml")
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
            sets.major.source  # for linter

    def test_set_settings_invalid(self):
        sets: Settings = Settings()
        # model is frozen, otherwise would raise no error
        with self.assertRaises(ValueError):
            sets.minor.model = "Gemini/gemini-latest"
            sets.minor.model

    def test_default_config(self):
        create_default_config_file("temp_config.toml")
        default_sets = load_settings("temp_config.toml")
        sets = Settings(
            **{'major': {'model': "Mistral/mistral-large-latest"}}
        )
        export_settings(sets, "temp_config.toml")
        create_default_config_file("temp_config.toml")
        sets = load_settings("temp_config.toml")
        self.assertEqual(
            sets.major.get_model_source(),
            default_sets.major.get_model_source(),
        )
        os.unlink("temp_config.toml")


if __name__ == "__main__":
    unittest.main()
