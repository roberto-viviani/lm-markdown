"""Test lazy dict"""

# pyright: basic

# intentional errors
# pyright: reportArgumentType=false

import unittest

from pydantic import validate_call

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


class CreateNewDictionary(unittest.TestCase):

    def test_create_dict(self):
        from typing import Literal
        from pydantic import BaseModel, ConfigDict, ValidationError
        from lmm.language_models.lazy_dict import LazyLoadingDict

        # This defines source + model
        class LanguageModelSpecification(BaseModel):
            source_name: Literal[
                'Anthropic', 'Gemini', 'Mistral', 'OpenAI'
            ]
            model_name: str

            # This required to make instances hashable, so that they can
            # be used as keys in the dictionary
            model_config = ConfigDict(frozen=True)

        # Langchain model type specified here.
        def _create_model_instance(
            src,
        ) -> LanguageModelSpecification:
            return LanguageModelSpecification(
                source_name=src, model_name="test"
            )

        lzd = LazyLoadingDict(_create_model_instance)

        obj = lzd['Anthropic']
        self.assertIsInstance(obj, LanguageModelSpecification)
        with self.assertRaises(ValidationError):
            obj = lzd['Kntropix']
        # adding object directly
        with self.assertRaises(ValidationError):
            obj = lzd['Kntropix'] = LanguageModelSpecification(
                source_name="Kntropix", model_name="test"
            )
        # Kntropix raises validation error here
        with self.assertRaises(ValidationError):
            obj = lzd['Gemini'] = LanguageModelSpecification(
                source_name="Kntropix", model_name="test"
            )
        # this, however, is ok
        obj = lzd['Mistral'] = LanguageModelSpecification(
            source_name="Gemini", model_name="test"
        )
        self.assertIsInstance(obj, LanguageModelSpecification)
        # this is also ok because python per se does not constrain
        # literals here
        obj = lzd['Zntropix'] = LanguageModelSpecification(
            source_name="Anthropic", model_name="test"
        )
        self.assertIsInstance(obj, LanguageModelSpecification)

    def test_create_constrained_dict(self):
        from pydantic import BaseModel, ConfigDict, ValidationError
        from lmm.language_models.lazy_dict import LazyLoadingDict
        from enum import StrEnum

        class LanguageModelSource(StrEnum):
            Anthropic = "Anthropic"
            Gemini = "Gemini"
            Mistral = "Mistral"
            OpenAI = "OpenAI"

        # This defines source + model
        class LanguageModelSpecification(BaseModel):
            source_name: LanguageModelSource
            model_name: str

            # This required to make instances hashable, so that they can
            # be used as keys in the dictionary
            model_config = ConfigDict(frozen=True)

        # Langchain model type specified here.
        def _create_model_instance(
            src: LanguageModelSource,
        ) -> LanguageModelSpecification:
            return LanguageModelSpecification(
                source_name=src, model_name="test"
            )

        lzd = LazyLoadingDict(_create_model_instance)

        obj = lzd['Anthropic']
        self.assertIsInstance(obj, LanguageModelSpecification)
        with self.assertRaises(ValidationError):
            obj = lzd['Kntropix']
        # adding object directly
        with self.assertRaises(ValidationError):
            obj = lzd['Kntropix'] = LanguageModelSpecification(
                source_name="Kntropix", model_name="test"
            )
        # Kntropix raises validation error here
        with self.assertRaises(ValidationError):
            obj = lzd['Gemini'] = LanguageModelSpecification(
                source_name="Kntropix", model_name="test"
            )
        # this, however, is ok
        obj = lzd['Mistral'] = LanguageModelSpecification(
            source_name="Gemini", model_name="test"
        )
        self.assertIsInstance(obj, LanguageModelSpecification)
        # this is also ok
        obj = lzd['Zntropix'] = LanguageModelSpecification(
            source_name="Anthropic", model_name="test"
        )
        self.assertIsInstance(obj, LanguageModelSpecification)

    def test_create_unconstrained_dict(self):
        from pydantic import BaseModel, ConfigDict
        from lmm.language_models.lazy_dict import LazyLoadingDict

        # This defines model
        class LanguageModelSpecification(BaseModel):
            source_name: str
            model_name: str

            # This required to make instances hashable, so that they can
            # be used as keys in the dictionary
            model_config = ConfigDict(frozen=True)

        # Langchain model type specified here.
        def _create_model_instance(
            src: str,
        ) -> LanguageModelSpecification:
            return LanguageModelSpecification(
                source_name=src, model_name="test"
            )

        lzd = LazyLoadingDict(_create_model_instance)

        obj = lzd['Anthropic']
        self.assertIsInstance(obj, LanguageModelSpecification)
        obj = lzd['Kntropix']
        self.assertIsInstance(obj, LanguageModelSpecification)
        # test adding object directly
        obj = lzd['Fututropic'] = LanguageModelSpecification(
            source_name="Fututropic", model_name="test"
        )
        self.assertIsInstance(obj, LanguageModelSpecification)
        obj = lzd['Zntropix'] = LanguageModelSpecification(
            source_name="Anthropic", model_name="test"
        )
        self.assertIsInstance(obj, LanguageModelSpecification)


class TestDestructor(unittest.TestCase):
    def test_explicit_destructor(self):
        destroyed = []

        def destructor(val):
            destroyed.append(val)

        lzd = LazyLoadingDict(lambda k: f"value_{k}", destructor_func=destructor)
        
        # Test __delitem__
        val = lzd["key1"]
        self.assertEqual(val, "value_key1")
        del lzd["key1"]
        self.assertIn("value_key1", destroyed)
        
        # Test clear
        lzd["key2"] = "value_key2"
        lzd.clear()
        self.assertIn("value_key2", destroyed)

    def test_protocol_destructor(self):
        class Disposable:
            def __init__(self, name):
                self.name = name
                self.closed = False
            
            def close(self):
                self.closed = True

        lzd = LazyLoadingDict(lambda k: Disposable(k))
        
        # Test __delitem__
        obj = lzd["key1"]
        del lzd["key1"]
        self.assertTrue(obj.closed)

    def test_overwrite_error(self):
        lzd = LazyLoadingDict(lambda k: k)
        lzd["key1"] = "val1"
        
        with self.assertRaises(ValueError):
            lzd["key1"] = "val2"

    def test_pop_no_destructor(self):
        destroyed = []
        lzd = LazyLoadingDict(lambda k: k, destructor_func=lambda x: destroyed.append(x))
        
        lzd["key1"]
        val = lzd.pop("key1")
        self.assertEqual(val, "key1")
        self.assertEqual(destroyed, [])

    def test_del_cleanup(self):
        import gc
        destroyed = []
        
        # Use a separate scope to ensure lzd is collected
        def create_and_destroy():
            lzd = LazyLoadingDict(lambda k: k, destructor_func=lambda x: destroyed.append(x))
            lzd["key1"]
            lzd["key2"]
            # lzd goes out of scope here
            
        create_and_destroy()
        gc.collect()
        
        self.assertIn("key1", destroyed)
        self.assertIn("key2", destroyed)


if __name__ == "__main__":
    unittest.main()
