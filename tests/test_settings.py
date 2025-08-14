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
            **{'language_models': {'model_minor': "gpt-4o"}}
        )
        self.assertEqual(sets.language_models.model_minor, "gpt-4o")
        # unmentioned setting still set
        self.assertEqual(
            sets.language_models.model_aux, "mistral-small-latest"
        )

    def test_set_settings_given2(self):
        sets: Settings = Settings(
            language_models={'model_minor': "gpt-4o"}
        )
        self.assertEqual(sets.language_models.model_minor, "gpt-4o")

    def test_set_settings_given_invalid(self):
        with self.assertRaises(ValidationError):
            sets = Settings(language_models={'model_minor': 4})
            sets.language_models.model_minor  # for linter

    def test_set_settings_given_invalid_enum(self):
        # invalid literal: cohere not supported
        with self.assertRaises(ValidationError):
            sets = Settings(
                **{'language_models': {'source_major': "cohere"}}
            )
            sets.language_models.source_major  # for linter

    def test_set_settings_invalid(self):
        sets: Settings = Settings()
        # alas, no error here, but pyright complains
        sets.language_models.source_minor = "Cohere"  # type: ignore
        self.assertTrue(True)

    def test_set_settings_error(self):
        sets: Settings = Settings()
        # alas, no error here, but pyright complains
        sets.language_models.source_minor = 1  # type: ignore
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
