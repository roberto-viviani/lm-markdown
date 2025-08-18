"""Test model selection for Langchain"""

import unittest

from lmm.language_models.langchain.models import (
    langchain_factory,
    langchain_embeddings,
    create_model_from_spec,
    create_model_from_settings,
    create_embedding_model_from_spec,
    LanguageModelSettings,
)

from pydantic import ValidationError


# reset at end of testing
def reset_langchain_factory():
    """Reset the langchain factory to empty"""
    langchain_factory.clear()
    langchain_embeddings.clear()


# Reset factory after running all tests
unittest.TestCase.setUpClass = reset_langchain_factory
unittest.TestCase.tearDownClass = reset_langchain_factory


class TestLazyDict(unittest.TestCase):

    def test_hashability(self):
        settings1 = LanguageModelSettings(
            model="OpenAI/gpt-4o-mini",
            provider_params={"top_p": 0.9},
        )
        settings2 = LanguageModelSettings(
            model="OpenAI/gpt-4o-mini",
            provider_params={"top_p": 0.9},
        )
        settings3 = LanguageModelSettings(
            model="OpenAI/gpt-4o-mini",
            provider_params={"top_p": 0.8},  # Different value
        )

        hash1 = hash(settings1)
        hash2 = hash(settings2)
        hash3 = hash(settings3)

        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertNotEqual(hash2, hash3)

        # Test using as dict key (what LazyLoadingDict does)
        test_dict = {settings1: "model1", settings3: "model3"}
        self.assertEqual(test_dict[settings1], "model1")
        self.assertEqual(test_dict[settings3], "model3")

    def test_create(self):
        model_spec = LanguageModelSettings(model="OpenAI/gpt-4o")
        model = langchain_factory[model_spec]
        self.assertEqual(model.get_name(), "ChatOpenAI")
        model_count = len(langchain_factory)

        # previously cached
        model_spec = {'model': "OpenAI/gpt-4o"}
        model = create_model_from_spec(**model_spec)
        self.assertEqual(model.get_name(), "ChatOpenAI")
        self.assertEqual(len(langchain_factory), model_count)

    def test_create3(self):
        # previously cached
        model_spec = {'model': "OpenAI/gpt-4o"}
        model = create_model_from_settings(
            LanguageModelSettings(**model_spec)
        )
        self.assertEqual(model.get_name(), "ChatOpenAI")
        model_count = len(langchain_factory)

        # new model
        model_spec = {'model': "OpenAI/gpt-4o", 'temperature': 0.7}
        model = create_model_from_settings(
            LanguageModelSettings(**model_spec)
        )
        self.assertEqual(model.get_name(), "ChatOpenAI")
        self.assertEqual(len(langchain_factory), model_count + 1)

        model_spec = {
            'model': "Anthropic/claude-3-5-haiku-latest",
        }
        model = create_model_from_spec(**model_spec)
        self.assertEqual(model.get_name(), "ChatAnthropic")
        self.assertEqual(len(langchain_factory), model_count + 2)

    def test_create_novel2(self):
        model_spec = {
            'model': "OpenAI/gpt-4o-mini",
        }
        model = create_model_from_spec(**model_spec)
        self.assertEqual(model.get_name(), "ChatOpenAI")

    def test_create_variant1(self):
        model_spec = {
            'model': "OpenAI/gpt-4o-mini",
            'temperature': 0.7,
            'max_retries': 6,
        }
        model = create_model_from_spec(**model_spec)
        self.assertEqual(model.get_name(), "ChatOpenAI")

    def test_create_variant1_obj(self):
        model_spec = {
            'model': "OpenAI/gpt-4o-mini",
            'temperature': 0.7,
            'max_retries': 6,
        }
        sets = LanguageModelSettings(**model_spec)
        model = create_model_from_settings(sets)
        self.assertEqual(model.get_name(), "ChatOpenAI")

    def test_create_variant1_call(self):
        model = create_model_from_spec(
            model="OpenAI/gpt-4o", temperature=0.7, max_retries=6
        )
        self.assertEqual(model.get_name(), "ChatOpenAI")

    def test_create_complete_spec(self):
        # Create a settings object with custom parameters
        settings = LanguageModelSettings(
            model="OpenAI/gpt-4o-mini",
            temperature=0.7,
            max_tokens=500,
            max_retries=3,
            timeout=30.0,
            provider_params={
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
                "top_p": 0.9,
                "seed": 42,
            },
        )
        model = create_model_from_settings(settings)
        self.assertEqual(getattr(model, 'temperature', 'N/A'), 0.7)
        self.assertEqual(
            getattr(model, 'frequency_penalty', 'N/A'), 0.1
        )
        self.assertEqual(
            getattr(model, 'presence_penalty', 'N/A'), 0.2
        )
        self.assertEqual(getattr(model, 'top_p', 'N/A'), 0.9)
        self.assertEqual(getattr(model, 'seed', 'N/A'), 42)

    def test_create_complete_spec_invalid(self):
        # Create a settings object with invalid custom parameters
        with self.assertRaises(ValueError):
            settings = LanguageModelSettings(
                model="OpenAI/gpt-4o-mini",
                temperature=0.7,
                max_tokens=500,
                max_retries=3,
                timeout=30.0,
                provider_params={
                    "frequency_penalty": 0.1,
                    "seed": 42,
                    "invalid_param": 0,
                },
            )
            model = create_model_from_settings(settings)
            model.get_name()

    def test_create_invalid(self):
        model_spec = {'model': "OpenX/gpt-4o"}
        with self.assertRaises(ValueError):
            model = create_model_from_spec(**model_spec)
            model.get_name()

    def test_create_invalid2(self):
        with self.assertRaises(ValueError):
            model_spec = LanguageModelSettings(model="OpenX/gpt-4o")
            str(model_spec)  # for linter

    def test_create_invalid3(self):
        with self.assertRaises(ValidationError):
            model = create_model_from_spec(None)
            model

    def test_create_embeddings(self):
        model_spec = {
            'dense_model': "OpenAI/text-embedding-3-small",
        }
        model = create_embedding_model_from_spec(**model_spec)
        self.assertEqual(len(langchain_embeddings), 1)
        self.assertEqual(model.__class__.__name__, "OpenAIEmbeddings")

    def test_create_embeddings2(self):
        model_spec = {
            'dense_model': "OpenAI/text-embedding-3-large",
        }
        model = create_embedding_model_from_spec(**model_spec)
        self.assertEqual(len(langchain_embeddings), 2)
        self.assertEqual(model.__class__.__name__, "OpenAIEmbeddings")

    def test_create_embeddings_invalid(self):
        with self.assertRaises(TypeError):
            model_spec = {
                'sparse_model': "OpenAI/text-embedding-3-large",
            }
            model = create_embedding_model_from_spec(**model_spec)
            model

    def test_create_embeddings_invalid2(self):
        with self.assertRaises(TypeError):
            model_spec = {
                'sparse_concept': "OpenAI/text-embedding-3-large",
            }
            model = create_embedding_model_from_spec(**model_spec)
            model


if __name__ == "__main__":
    unittest.main()
