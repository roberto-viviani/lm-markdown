"""Test language_models.tools"""

import unittest

from lmm.language_models.tools import (
    tool_library as kernel_prompts,
    create_prompt,
    ToolDefinition,
)
from lmm.config.config import Settings, export_settings

original_settings = Settings()


def setUpModule():
    settings = Settings(
        major={'model': "Debug/debug"},
        minor={'model': "Debug/debug"},
        aux={'model': "Debug/debug"},
    )
    export_settings(settings)


def tearDownModule():
    export_settings(original_settings)


class TestKernelPrompts(unittest.TestCase):

    def test_setup(self):
        settings = Settings()
        self.assertEqual(settings.major.get_model_source(), "Debug")
        self.assertEqual(settings.minor.get_model_source(), "Debug")
        self.assertEqual(settings.aux.get_model_source(), "Debug")
        self.assertEqual(settings.major.get_model_name(), "debug")
        self.assertEqual(settings.minor.get_model_name(), "debug")
        self.assertEqual(settings.aux.get_model_name(), "debug")

    def test_get_kernel(self):
        prompt: str = kernel_prompts["summarizer"]
        self.assertTrue(bool(prompt))

    def test_invalid_kernel_name(self):
        with self.assertRaises(ValueError):
            prompt: str = kernel_prompts["bonobo"]
            prompt


class TestCustomPrompt(unittest.TestCase):

    def test_custom_prompt(self):
        prompt_template = """
Provide the questions to which the text answers.

TEXT:
{text}
"""
        create_prompt(prompt_template, "questioner")
        prompt: ToolDefinition = kernel_prompts["questioner"]
        self.assertEqual(prompt.prompt, prompt_template)


if __name__ == "__main__":
    unittest.main()
