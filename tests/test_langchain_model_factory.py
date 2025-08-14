"""Test model selection for Langchain"""

import unittest

from lmm.language_models.langchain.model_factory import (
    langchain_factory,
    langchain_embeddings,
    create_model_from_spec,
    create_embedding_model_from_spec,
    LanguageModelSettings,
)


# reset at end of testing
def reset_langchain_factory():
    """Reset the langchain factory to empty"""
    langchain_factory.clear()
    langchain_embeddings.clear()


# Reset factory after running all tests
unittest.TestCase.setUpClass = reset_langchain_factory
unittest.TestCase.tearDownClass = reset_langchain_factory


class TestLazyDict(unittest.TestCase):

    def test_create(self):
        model_spec = LanguageModelSettings(
            source="OpenAI", name_model="gpt-4o"
        )
        model = langchain_factory[model_spec]
        self.assertEqual(model.get_name(), "ChatOpenAI")
        self.assertEqual(len(langchain_factory), 1)

    def test_create2(self):
        # previously cached
        model_spec = {'source_name': "OpenAI", 'model_name': "gpt-4o"}
        model = create_model_from_spec(**model_spec)
        self.assertEqual(model.get_name(), "ChatOpenAI")
        self.assertEqual(len(langchain_factory), 1)

    def test_create_novel(self):
        # previously cached
        model_spec = {
            'source_name': "Anthropic",
            'model_name': "'claude-3-5-haiku-latest'",
        }
        model = create_model_from_spec(**model_spec)
        self.assertEqual(model.get_name(), "ChatAnthropic")
        self.assertEqual(len(langchain_factory), 2)

    def test_create_novel2(self):
        # previously cached
        model_spec = {
            'source_name': "OpenAI",
            'model_name': "gpt-4o-mini",
        }
        model = create_model_from_spec(**model_spec)
        self.assertEqual(model.get_name(), "ChatOpenAI")
        self.assertEqual(len(langchain_factory), 3)

    def test_create_invalid(self):
        model_spec = {'source_name': "OpenX", 'model_name': "gpt-4o"}
        with self.assertRaises(ValueError):
            model = create_model_from_spec(**model_spec)
            model.get_name()

    def test_create_invalid2(self):
        with self.assertRaises(ValueError):
            model_spec = LanguageModelSettings(
                source="OpenX", name_model="gpt-4o"
            )
            str(model_spec)  # for linter

    def test_create_embeddings(self):
        model_spec = {
            'source_name': "OpenAI",
            'model_name': "text-embedding-3-small",
        }
        model = create_embedding_model_from_spec(**model_spec)
        self.assertEqual(len(langchain_embeddings), 1)
        self.assertEqual(model.__class__.__name__, "OpenAIEmbeddings")

    def test_create_embeddings2(self):
        model_spec = {
            'source_name': "OpenAI",
            'model_name': "text-embedding-3-large",
        }
        model = create_embedding_model_from_spec(**model_spec)
        self.assertEqual(len(langchain_embeddings), 2)
        self.assertEqual(model.__class__.__name__, "OpenAIEmbeddings")


if __name__ == "__main__":
    unittest.main()
