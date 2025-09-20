"""Test language_models.tools"""

import unittest

from lmm.language_models.tools import (
    tool_library as kernel_prompts,
    create_prompt,
    ToolDefinition,
)


class TestKernelPrompts(unittest.TestCase):

    def test_get_kernel(self):
        prompt: str = kernel_prompts["summarizer"]
        self.assertTrue(bool(prompt))

    def test_invalid_kernel_name(self):
        with self.assertRaises(ValueError):
            prompt: str = kernel_prompts["bonobo"]
            prompt


class TestCustomPrompt(unittest.TestCase):

    def test_custorm_prompt(self):
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
