"""Test settings module"""

# pyright: basic

import unittest
from lmm.settings.settings import Settings, serialize_settings
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
        self.assertEqual(sets.aux.name_model, "mistral-small-latest")

    def test_set_settings_given2(self):
        sets: Settings = Settings(minor={'name_model': "gpt-4o"})
        self.assertEqual(sets.minor.name_model, "gpt-4o")

    def test_set_settings_given_invalid(self):
        with self.assertRaises(ValidationError):
            sets = Settings(minor={'name_model': 4})
            sets.minor.name_model  # for linter

    def test_set_settings_given_invalid_enum(self):
        # invalid literal: cohere not supported
        with self.assertRaises(ValidationError):
            sets = Settings(**{'major': {'source': "cohere"}})
            sets.major.source  # for linter

    def test_set_settings_invalid(self):
        sets: Settings = Settings()
        # model is frozen, otherwise would raise no error
        with self.assertRaises(ValidationError):
            sets.minor.source = "Gemini"
            sets.minor.source


if __name__ == "__main__":
    unittest.main()
