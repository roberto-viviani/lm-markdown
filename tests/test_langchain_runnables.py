"""Test langchain kernels"""

# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportUnusedExpression=false

import os
import unittest


from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from lmm.models.langchain.runnables import (
    create_runnable as create_kernel,
    create_embeddings,
    create_kernel_from_objects,
)
from lmm.config.config import (
    Settings,
    LanguageModelSettings,
    EmbeddingSettings,
    export_settings,
)


# Add this near the top of the file, after other imports
OPENAI_KEY_AVAILABLE = os.environ.get("OPENAI_API_KEY") is not None

base_settings = Settings()


def _get_name(kn: str, sn: str, mn: str) -> str:
    return f"{kn}:{sn}/{mn}"

@unittest.skipUnless(OPENAI_KEY_AVAILABLE, "OpenAI API key not available")
class TestDefaultModels(unittest.TestCase):

    def test_query(self):
        model = create_kernel("query")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "query",
                base_settings.major.get_model_source(),
                base_settings.major.get_model_name(),
            ),
        )

    def test_query_with_context(self):
        model = create_kernel("query_with_context")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "query_with_context",
                base_settings.major.get_model_source(),
                base_settings.major.get_model_name(),
            ),
        )

    def test_summarizer(self):
        model = create_kernel("summarizer")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "summarizer",
                base_settings.minor.get_model_source(),
                base_settings.minor.get_model_name(),
            ),
        )

    def test_question_generator(self):
        model = create_kernel("question_generator")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "question_generator",
                base_settings.minor.get_model_source(),
                base_settings.minor.get_model_name(),
            ),
        )

    def test_context_validation(self):
        model = create_kernel("context_validator")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "context_validator",
                base_settings.aux.get_model_source(),
                base_settings.aux.get_model_name(),
            ),
        )

    def test_check_allowed_content_invalid(self):
        # Missing kernel params
        with self.assertRaises(ValueError):
            create_kernel("allowed_content_validator")

    def test_check_allowed_content(self):
        model = create_kernel(
            "allowed_content_validator",
            allowed_content=["statictics"],
        )
        self.assertEqual(
            model.get_name(),
            _get_name(
                "allowed_content_validator",
                base_settings.aux.get_model_source(),
                base_settings.aux.get_model_name(),
            ),
        )

    def test_set_model(self):
        model = create_kernel(
            "summarizer",
            {
                'model': "Mistral/mistral-small-latest",
                'temperature': 0.7,
            },
        )
        self.assertEqual(
            model.get_name(),
            _get_name(
                "summarizer",
                "Mistral",
                "mistral-small-latest",
            ),
        )

    def test_set_model_obj(self):
        model = create_kernel(
            "summarizer",
            LanguageModelSettings(
                model="Mistral/mistral-small-latest"
            ),
        )
        self.assertEqual(
            model.get_name(),
            _get_name(
                "summarizer",
                "Mistral",
                "mistral-small-latest",
            ),
        )

    def test_set_model_settings(self):
        model = create_kernel(
            "summarizer",
            Settings(minor={'model': "Mistral/mistral-small-latest"}),
        )
        self.assertEqual(
            model.get_name(),
            _get_name(
                "summarizer",
                "Mistral",
                "mistral-small-latest",
            ),
        )

    def test_set_invalid_model(self):
        with self.assertRaises(ValueError):
            model = create_kernel(
                "summarizer",
                {
                    'model': "Cohere/cohere-small-latest",
                },
            )
            model.get_name()

    def test_set_invalid_kernel(self):
        with self.assertRaises(ValueError):
            model = create_kernel(
                "joker",
                {
                    'model': "Mistral/mistral-small-latest",
                },
            )
            model.get_name()

    def test_custom_model(self):
        from lmm.models.prompts import (
            prompt_library as kernel_prompts,
            create_prompt,
        )

        prompt_template = """
Provide the questions to which the text answers.

TEXT:
{text}
"""
        create_prompt(name="questioner2", prompt=prompt_template)
        prompt = kernel_prompts["questioner2"]
        self.assertEqual(prompt.prompt, prompt_template)

        settings = Settings()
        model = create_kernel("questioner2", settings.major)
        self.assertIn(
            f"{base_settings.major.get_model_source()}/"
            + f"{base_settings.major.get_model_name()}",
            model.get_name(),
        )

    def test_custom_model_from_config(self):
        from lmm.models.prompts import (
            prompt_library as kernel_prompts,
            create_prompt,
        )

        prompt_template = """
Provide the questions to which the text answers.

TEXT:
{text}
"""
        kernel_prompts.clear()
        create_prompt(name="questioner", prompt=prompt_template)
        prompt = kernel_prompts["questioner"]
        self.assertEqual(prompt.prompt, prompt_template)

        model = create_kernel("questioner")
        self.assertIn(
            f"{base_settings.minor.get_model_source()}/"
            + f"{base_settings.minor.get_model_name()}",
            model.get_name(),
        )

    def test_create_kernel_from_objects(self):
        human_prompt = "Tell me a story"
        settings = Settings()
        model = create_kernel_from_objects(
            human_prompt=human_prompt,
            system_prompt="You are a funny storyteller",
            language_model=settings.aux,
        )
        self.assertIn(
            f"{settings.aux.get_model_source()}/"
            + f"{settings.aux.get_model_name()}",
            model.get_name(),
        )

    def test_create_kernel_from_objects2(self):

        prompt_template = """
Provide the questions to which the text answers.

TEXT:
{text}
"""
        settings = Settings()
        model = create_kernel_from_objects(
            human_prompt=prompt_template,
            system_prompt="You are a university tutor",
            language_model=settings.aux,
        )
        self.assertIn(
            f"{settings.aux.get_model_source()}/"
            + f"{settings.aux.get_model_name()}",
            model.get_name(),
        )


class TestDebugModel(unittest.TestCase):

    # detup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = Settings()

    @classmethod
    def setUpClass(cls):
        settings = Settings(
            major={'model': "Debug/debug"},
            minor={'model': "Debug/debug"},
            aux={'model': "Debug/debug"},
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def test_setup(self):
        settings = Settings()
        self.assertEqual(settings.major.get_model_source(), "Debug")
        self.assertEqual(settings.minor.get_model_source(), "Debug")
        self.assertEqual(settings.aux.get_model_source(), "Debug")
        self.assertEqual(settings.major.get_model_name(), "debug")
        self.assertEqual(settings.minor.get_model_name(), "debug")
        self.assertEqual(settings.aux.get_model_name(), "debug")

    def test_debug_kernel(self):
        model = create_kernel(
            runnable_name="summarizer",
            user_settings={
                'model': "Debug/debug",
            },
        )
        msg = model.invoke({'text': "This is test text"})
        self.assertEqual(msg, "This is a summary of the text.")

@unittest.skipUnless(OPENAI_KEY_AVAILABLE, "OpenAI API key not available")
class TestEmbeddingModel(unittest.TestCase):

    # setup and teardown replace config.toml to avoid
    # calling the language model server
    original_settings = Settings()

    @classmethod
    def setUpClass(cls):
        settings = Settings(
            embeddings={
                'dense_model': "OpenAI/text-embedding-3-small"
            }
        )
        export_settings(settings)

    @classmethod
    def tearDownClass(cls):
        settings = cls.original_settings
        export_settings(settings)

    def test_embedding_default(self) -> None:
        embmodel = create_embeddings()
        self.assertIsInstance(embmodel, Embeddings)

    def test_spec_embedding(self) -> None:
        embmodel = create_embeddings(
            EmbeddingSettings(
                **{'dense_model': "OpenAI/text-embedding-3-small"}
            )
        )
        self.assertIsInstance(embmodel, Embeddings)

    def test_spec_embedding_dict(self) -> None:
        embmodel = create_embeddings(
            {'dense_model': "OpenAI/text-embedding-3-small"}
        )
        self.assertIsInstance(embmodel, Embeddings)

    def test_spec_embedding_settings(self) -> None:
        embmodel = create_embeddings(
            Settings(
                embeddings={
                    'dense_model': "OpenAI/text-embedding-3-small"
                }
            )
        )
        self.assertIsInstance(embmodel, Embeddings)

    # sentence transformers now installed.
    # def test_spec_embedding_import(self) -> None:
    #     with self.assertRaises(ImportError):
    #         create_embeddings(
    #             {
    #                 'dense_model': "SentenceTransformers/all-mpnet-base-v2"
    #             }
    #         )

    def test_invalid_spec_embedding(self) -> None:
        with self.assertRaises(ValidationError):
            create_embeddings(
                EmbeddingSettings(**{'dense_model': "Pippo/this"})
            )

    def test_invalid_spec_embedding_dict(self) -> None:
        with self.assertRaises(ValidationError):
            create_embeddings({'dense_model': "Pippo/this"})

    def test_invalid_spec_embedding2(self) -> None:
        with self.assertRaises(ValidationError):
            create_embeddings(
                EmbeddingSettings(
                    **{'densest': "OpenAI/text-embedding-3-small"}
                )
            )

    def test_invalid_spec_embedding2_dict(self) -> None:
        with self.assertRaises(ValidationError):
            create_embeddings(
                {'densest': "OpenAI/text-embedding-3-small"}
            )

    def test_create_embedding(self) -> None:
        emodel: Embeddings = create_embeddings(
            {'dense_model': "OpenAI/text-embedding-3-small"}
        )
        emodel
        # this tests ok, de-comment to reactivate
        # vect = emodel.embed_query("This sentence to embed")
        # self.assertIsInstance(vect, list)
        # self.assertIsInstance(vect[0], float)


if __name__ == "__main__":
    unittest.main()
