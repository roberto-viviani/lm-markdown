"""Test lazy dict"""

import unittest

from enum import StrEnum
from lmm.language_models.lazy_dict import (
    LazyLoadingDict,
)


# This is the class we store in the lazy dict.
class ModelClass:
    model_name: str

    def __init__(self, model_name):
        self.model_name = model_name

    def info(self) -> str:
        return str(self.model_name)


# We define here permissible keys in the lazy dict
class LMSource(StrEnum):
    Anthropic = 'Anthropic'
    Gemini = 'Gemini'
    OpenAI = 'OpenAI'


# A factory function that creates a model class using the permissible key
# as an info for its creation.
from pydantic import validate_call


@validate_call
def create_model_instance(model_name: LMSource) -> ModelClass:
    """A placeholder function to simulate a data creation process.
    Throws ValueError if invalid model_name"""
    print(f"Created instance of {str(model_name)}")
    return ModelClass(model_name=model_name)


class TestLazyDict(unittest.TestCase):

    func_factory = LazyLoadingDict(create_model_instance)

    def test_dict(self):
        # prints 'Created instance of Gemini'
        model = self.func_factory[LMSource("Gemini")]
        self.assertEqual(model.info(), "Gemini")

    def test_dict2(self):
        # Prints nothing
        model = self.func_factory[LMSource("Gemini")]
        self.assertEqual(model.info(), "Gemini")

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            model = self.func_factory[LMSource("OpenX")]
            model.info()


if __name__ == "__main__":
    unittest.main()
