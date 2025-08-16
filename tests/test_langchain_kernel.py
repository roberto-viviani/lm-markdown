"""Test langchain kernels"""

import unittest

from lmm.language_models.langchain.kernel import create_kernel
from lmm.config.config import Settings, LanguageModelSettings

base_settings = Settings()


def _get_name(kn: str, sn: str, mn: str) -> str:
    return f"{kn}:{sn}/{mn}"


class TestDefaultModels(unittest.TestCase):

    def test_query(self):
        model = create_kernel("query")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "query",
                base_settings.major.source,
                base_settings.major.name_model,
            ),
        )

    def test_query_with_context(self):
        model = create_kernel("query_with_context")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "query_with_context",
                base_settings.major.source,
                base_settings.major.name_model,
            ),
        )

    def test_summarizer(self):
        model = create_kernel("summarizer")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "summarizer",
                base_settings.minor.source,
                base_settings.minor.name_model,
            ),
        )

    def test_question_generator(self):
        model = create_kernel("question_generator")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "question_generator",
                base_settings.minor.source,
                base_settings.minor.name_model,
            ),
        )

    def test_check_content(self):
        model = create_kernel("check_content")
        self.assertEqual(
            model.get_name(),
            _get_name(
                "check_content",
                base_settings.aux.source,
                base_settings.aux.name_model,
            ),
        )

    def test_set_model(self):
        model = create_kernel(
            "summarizer",
            {
                'source': "Mistral",
                'name_model': "mistral-small-latest",
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
                source="Mistral", name_model="mistral-small-latest"
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

    def test_set_invalid_model(self):
        with self.assertRaises(ValueError):
            model = create_kernel(
                "summarizer",
                {
                    'source': "Cohere",
                    'name_model': "cohere-small-latest",
                },
            )
            model.get_name()

    def test_set_invalid_kernel(self):
        with self.assertRaises(ValueError):
            model = create_kernel(
                "joker",
                {
                    'source': "Mistral",
                    'name_model': "mistral-small-latest",
                },
            )
            model.get_name()


if __name__ == "__main__":
    unittest.main()
