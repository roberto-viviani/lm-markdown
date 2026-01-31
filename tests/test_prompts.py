"""Test language_models.tools"""

# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportUnusedExpression=false

import unittest

from lmm.language_models.prompts import (
    prompt_library as kernel_prompts,
    create_prompt,
    PromptDefinition,
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
        prompt: PromptDefinition = kernel_prompts["summarizer"]
        self.assertTrue(bool(prompt))

    def test_invalid_kernel_name(self):
        with self.assertRaises(ValueError):
            prompt: PromptDefinition = kernel_prompts["bonobo"]
            prompt


class TestCustomPrompt(unittest.TestCase):

    def test_custom_prompt(self):
        prompt_template = """
Provide the questions to which the text answers.

TEXT:
{text}
"""
        create_prompt(prompt_template, "new_prompt")
        prompt: PromptDefinition = kernel_prompts["new_prompt"]
        self.assertEqual(prompt.prompt, prompt_template)

    def test_repeated_custorm_prompt(self):
        prompt_template = """Will be rejected."""

        # cannot register another prompt with same name
        with self.assertRaises(ValueError):
            create_prompt(prompt_template, "new_prompt")

    def test_replaced_custom_prompt(self):
        prompt_template = """
Provide a different to which the text answers.

TEXT:
{text}
"""
        # register prompt with same name but replace set
        create_prompt(prompt_template, "new_prompt", replace=True)
        prompt: PromptDefinition = kernel_prompts["new_prompt"]
        self.assertEqual(prompt.prompt, prompt_template)
        

if __name__ == "__main__":
    unittest.main()
