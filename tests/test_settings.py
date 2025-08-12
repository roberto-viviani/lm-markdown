"""Test settings module"""

# pyright: basic

import unittest
from lmm.settings.settings import Settings, serialize_settings


class TestSettings(unittest.TestCase):
    def test_create_settings(self):
        sets: Settings = Settings()
        conf: str = serialize_settings(sets)
        self.assertTrue(bool(conf))

    def test_set_settings(self):
        sets: Settings = Settings()
        sets.language_models.model_minor = "gpt-4o-mini"
        self.assertEqual(
            sets.language_models.model_minor, "gpt-4o-mini"
        )

    def test_set_settings_invalid(self):
        sets: Settings = Settings()
        # alas, no error here
        sets.language_models.source_minor = "Cohere"  # type: ignore

    def test_set_settings_error(self):
        sets: Settings = Settings()
        # alas, no error here
        sets.language_models.source_minor = 1  # type: ignore


if __name__ == "__main__":
    unittest.main()
