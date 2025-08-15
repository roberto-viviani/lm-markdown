"""Test language_models.kernels"""

import unittest

from lmm.language_models.kernels import kernel_prompts


class TestKernelPrompts(unittest.TestCase):

    def test_get_kernel(self):
        prompt: str = kernel_prompts["summarizer"]
        self.assertTrue(bool(prompt))

    def test_invalid_kernel_name(self):
        with self.assertRaises(ValueError):
            prompt: str = kernel_prompts["bonobo"]
            prompt


if __name__ == "__main__":
    unittest.main()
