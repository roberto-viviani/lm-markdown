"""Test settings module"""

import unittest
import os

from lmm.config.config import (
    Settings,
    serialize_settings,
    export_settings,
    LanguageModelSettings,
    load_settings,
)
from pydantic_core import ValidationError


class TestSettings(unittest.TestCase):
    def test_create_settings(self):
        sets: Settings = Settings()
        conf: str = serialize_settings(sets)
        self.assertTrue(bool(conf))

    def test_set_settings_given(self):
        sets: Settings = Settings(
            **{'minor': {'source': "OpenAI", 'name_model': "gpt-4o"}}
        )
        self.assertEqual(sets.minor.name_model, "gpt-4o")
        # unmentioned setting still set
        self.assertEqual(sets.aux.source, "Mistral")
        self.assertEqual(sets.aux.name_model, "mistral-small-latest")

    def test_set_settings_given2(self):
        sets: Settings = Settings(minor={'name_model': "gpt-4o-nano"})
        self.assertEqual(sets.minor.name_model, "gpt-4o-nano")
        # unmentioned setting still set
        self.assertEqual(sets.aux.source, "Mistral")
        self.assertEqual(sets.aux.name_model, "mistral-small-latest")

    def test_set_settings_given3(self):
        sets: Settings = Settings(
            minor=LanguageModelSettings(
                **{'name_model': "gpt-4o-xxx"}
            )
        )
        self.assertEqual(sets.minor.name_model, "gpt-4o-xxx")
        # unmentioned setting still set
        self.assertEqual(sets.aux.source, "Mistral")
        self.assertEqual(sets.aux.name_model, "mistral-small-latest")

    def test_set_settings_incomplete(self):
        # need to specify name model
        with self.assertRaises(ValueError):
            sets: Settings = Settings(
                **{'minor': {'source': "OpenAI"}}
            )
            sets

    def test_readwrite_settings(self):
        set = Settings(
            major={'source': "Gemini", 'name_model': "gemini-latest"}
        )
        export_settings(set, "config_test.toml")

        sets: Settings = load_settings("config_test.toml")
        # expected result: name_model as in file
        self.assertEqual(sets.major.source, "Gemini")
        self.assertEqual(set.major.name_model, "gemini-latest")
        # unmentioned setting still set
        self.assertEqual(sets.aux.source, "Mistral")
        self.assertEqual(sets.aux.name_model, "mistral-small-latest")
        os.unlink("config_test.toml")

    def test_set_settings_given_invalid(self):
        with self.assertRaises(ValidationError):
            sets = Settings(minor={'name_model': 4})
            sets.minor.name_model  # for linter

    def test_set_settings_given_invalid_enum(self):
        # invalid literal: cohere not supported
        with self.assertRaises(ValidationError):
            sets = Settings(
                **{
                    'major': {
                        'source': "cohere",
                        'name_model': "latest",
                    }
                }
            )
            sets.major.source  # for linter

    def test_set_settings_invalid(self):
        sets: Settings = Settings()
        # model is frozen, otherwise would raise no error
        with self.assertRaises(ValidationError):
            sets.minor.source = "Gemini"
            sets.minor.source


if __name__ == "__main__":
    unittest.main()
