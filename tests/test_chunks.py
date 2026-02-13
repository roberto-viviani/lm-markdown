"""tests for chunk.py"""

# pyright: basic
# pyright: reportMissingTypeStubs=false
# pyright: reportIndexIssue=false

import unittest

# from lmm_education.config.config import AnnotationModel
from lmm.scan.chunks import (
    blocks_to_chunks,
    chunks_to_blocks,
    EncodingModel,
    AnnotationModel,
)
from lmm.markdown.parse_markdown import (
    HeaderBlock,
    MetadataBlock,
    HeadingBlock,
    TextBlock,
    ErrorBlock,
    Block,
    parse_markdown_text,
)
from lmm.scan.scan_keys import (
    QUESTIONS_KEY,
    CHAT_KEY,
    SUMMARY_KEY,
    TITLES_KEY,
)
from lmm.scan.scan_rag import blocklist_rag, ScanOpts
from lmm.config.config import Settings, export_settings


header = HeaderBlock(content={'title': "Test blocklist"})
metadata = MetadataBlock(
    content={
        QUESTIONS_KEY: "What is the nature of the test? - How can we fix it?",
        CHAT_KEY: "Some discussion",
        SUMMARY_KEY: "The summary of the text.",
    }
)
heading = HeadingBlock(level=2, content="First title")
text = TextBlock(content="This is text following the heading")

blocks: list[Block] = [header, metadata, heading, text]
lenblocks: int = len(blocks)


class TestChunkNulls(unittest.TestCase):

    def test_empty_list(self):
        chunks = blocks_to_chunks([], EncodingModel.CONTENT)
        self.assertEqual(len(chunks), 0)

    def test_nontext_list(self):
        chunks = blocks_to_chunks(
            [header, metadata, heading], EncodingModel.CONTENT
        )
        self.assertEqual(len(chunks), 0)

    def test_dangling_metadata(self):
        # this creates an empty chunk list as there is no text
        # content.
        chunks = blocks_to_chunks(
            [header, metadata, heading, metadata],
            EncodingModel.CONTENT,
        )
        self.assertEqual(len(chunks), 0)


class TestBlocklistWithErrors(unittest.TestCase):

    def test_blocklist_errors(self):
        # expected behaviour: empty chunk list
        chunks = blocks_to_chunks(
            blocks + [ErrorBlock(content="Invalid block here")],
            EncodingModel.CONTENT,
        )
        self.assertEqual(len(chunks), 0)


class TestChunkFormation(unittest.TestCase):

    def test_transf_and_inverse(self) -> None:
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)
        reformed_blocks = chunks_to_blocks(chunks, "")
        # expected result: a metadata and a text block
        self.assertEqual(len(reformed_blocks), 2)
        self.assertIsInstance(reformed_blocks[0], MetadataBlock)
        self.assertIsInstance(reformed_blocks[1], TextBlock)

    def test_uuid_formation(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)
        uuids = [chunk.uuid for chunk in chunks]
        self.assertEqual(len(uuids), len(chunks))
        for u in uuids[1:]:
            self.assertIsInstance(u, str)
            self.assertNotEqual(uuids[0], u)

    def test_docid_unequal(self):
        from lmm.markdown.parse_markdown import blocklist_copy

        new_blocks: list[Block] = blocklist_copy(blocks)
        new_blocks[0].content['docid'] = "test1"
        chunks = blocks_to_chunks(new_blocks, EncodingModel.CONTENT)

        uuids1 = [chunk.uuid for chunk in chunks]

        new_blocks = blocklist_copy(blocks)
        new_blocks[0].content['docid'] = "test2"
        chunks = blocks_to_chunks(new_blocks, EncodingModel.CONTENT)

        uuids2 = [chunk.uuid for chunk in chunks]
        self.assertNotEqual(uuids1, uuids2)

    def test_docid_equal(self):
        from lmm.markdown.parse_markdown import blocklist_copy

        new_blocks = blocklist_copy(blocks)
        new_blocks[0].content['docid'] = "test1"
        chunks = blocks_to_chunks(new_blocks, EncodingModel.CONTENT)

        uuids1 = [chunk.uuid for chunk in chunks]

        new_blocks = blocklist_copy(blocks)
        new_blocks[0].content['docid'] = "test1"
        chunks = blocks_to_chunks(new_blocks, EncodingModel.CONTENT)

        uuids2 = [chunk.uuid for chunk in chunks]
        self.assertEqual(uuids1, uuids2)

    def test_docid_random(self):
        from lmm.markdown.parse_markdown import blocklist_copy

        new_blocks = blocklist_copy(blocks)
        chunks = blocks_to_chunks(new_blocks, EncodingModel.CONTENT)

        uuids1 = [chunk.uuid for chunk in chunks]

        new_blocks = blocklist_copy(blocks)
        chunks = blocks_to_chunks(new_blocks, EncodingModel.CONTENT)

        uuids2 = [chunk.uuid for chunk in chunks]
        self.assertNotEqual(uuids1, uuids2)


class TestChunkInheritance(unittest.TestCase):

    def test_annotate_questions(self):
        from lmm.scan.scan_rag import blocklist_rag, ScanOpts
        from lmm.markdown.parse_markdown import blocklist_copy

        self.assertEqual(len(blocks), lenblocks)
        blocklist = blocklist_rag(
            blocklist_copy(blocks),
            ScanOpts(titles=True, questions=True),
        )
        print(blocklist)
        annotation_model = AnnotationModel(
            inherited_properties=[QUESTIONS_KEY, TITLES_KEY],
        )
        chunks = blocks_to_chunks(
            blocklist, EncodingModel.CONTENT, annotation_model
        )
        self.assertEqual(len(blocklist), lenblocks)
        chunk = chunks[0]
        print(chunk.annotations)
        self.assertTrue(
            str(metadata.content[QUESTIONS_KEY]) in chunk.annotations
        )
        self.assertTrue(
            str("Test blocklist - First title") in chunk.annotations
        )

    def test_inherit_summary(self):
        self.assertEqual(len(blocks), lenblocks)
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)
        self.assertEqual(len(blocks), lenblocks)
        chunk = chunks[0]
        self.assertTrue(QUESTIONS_KEY in chunk.metadata)
        self.assertTrue(SUMMARY_KEY in chunk.metadata)


class TestChunkEncoding(unittest.TestCase):

    def test_encoding_NULL(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.NONE)
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, "")

    def test_encoding_CONTENT(self):
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.dense_encoding, text.get_content())

    def test_encoding_MERGED(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.MERGED, [QUESTIONS_KEY]
        )
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(
            chunk.dense_encoding,
            chunk.annotations + ". " + text.get_content(),
        )

    def test_encoding_SPARSE(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE, [QUESTIONS_KEY]
        )
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)

    def test_encoding_SPARSE_CONTENT(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT, [QUESTIONS_KEY]
        )
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)
        self.assertEqual(chunk.dense_encoding, text.get_content())

    def test_encoding_SPARSE_MERGED(self):
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_MERGED, [QUESTIONS_KEY]
        )
        self.assertEqual(len(blocks), lenblocks)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.sparse_encoding, chunk.annotations)
        self.assertEqual(
            chunk.dense_encoding,
            chunk.annotations + ". " + text.get_content(),
        )


class TestSkips(unittest.TestCase):
    # setup and teardown replace config.toml to avoid
    # calling the language model server
    original_config_content: str | None = None
    config_path = "config.toml"
    original_env: dict[str, str] = {}

    @classmethod
    def setUpClass(cls):
        # Backup existing config.toml content if it exists
        import os
        if os.path.exists(cls.config_path):
            with open(cls.config_path, "r", encoding="utf-8") as f:
                cls.original_config_content = f.read()
            os.remove(cls.config_path)

        # Clear potentially conflicting env vars
        cls.original_env = {}
        for key in list(os.environ.keys()):
            if key.startswith("LMM_"):
                cls.original_env[key] = os.environ.pop(key)

        try:
            # Create new settings (will use defaults + args since file is gone)
            settings = Settings(
                major={'model': "Debug/debug"}, # type: ignore
                minor={'model': "Debug/debug"}, # type: ignore
                aux={'model': "Debug/debug"}, # type: ignore
            )
            export_settings(settings)
        except Exception:
            # Restore config if settings creation fails
            if cls.original_config_content is not None:
                with open(cls.config_path, "w", encoding="utf-8") as f:
                    f.write(cls.original_config_content)
            # Restore env vars
            for key, val in cls.original_env.items():
                os.environ[key] = val
            raise

    @classmethod
    def tearDownClass(cls):
        import os
        # Remove the temporary test config
        if os.path.exists(cls.config_path):
            os.remove(cls.config_path)
            
        # Restore original config if it existed
        if cls.original_config_content is not None:
            with open(cls.config_path, "w", encoding="utf-8") as f:
                f.write(cls.original_config_content)

        # Restore env vars
        for key, val in cls.original_env.items():
            os.environ[key] = val

    def setUp(self):
        self.doc_with_skip = """
---
title: Document with skips
---

# Heading 1 (Process)

Text under heading 1.

---
skip: True
...
## Heading 2 (Skip)

Text under heading 2 that should be skipped.

### Heading 3 (Also Skip)

Text under heading 3.

# Heading 4 (Process)

Text under heading 4.

---
skip: True
...
This is a skipped text block.

# Heading 5 (Process)

Final text block.
"""
        self.blocks = parse_markdown_text(self.doc_with_skip)

    def test_skip_blocks(self):
        """Test that only non-skipped blocks are chunked."""
        blocks = blocklist_rag(self.blocks)
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)

        # only 2 blocks left
        self.assertEqual(len(chunks), 3)

    def test_notskipped_haveid(self):
        """Test all non-skipped chunked blocks have id"""
        from lmm.scan.scan_keys import TEXTID_KEY

        blocks = blocklist_rag(self.blocks)
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)
        for c in chunks:
            self.assertIn(TEXTID_KEY, c.metadata)

    def test_notskipped_havequestions(self):
        """Test all non-skipped chunked blocks have questions"""
        from lmm.scan.scan_keys import QUESTIONS_KEY

        blocks = blocklist_rag(
            self.blocks,
            ScanOpts(questions=True, questions_threshold=0),
        )
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT, [QUESTIONS_KEY]
        )
        for c in chunks:
            self.assertIn(QUESTIONS_KEY, c.metadata)

    def test_notskipped_havesummaries(self):
        """Test all non-skipped chunked blocks have summaries"""
        from lmm.scan.chunks import serialize_chunks
        from lmm.markdown.parse_markdown import serialize_blocks
        from lmm.scan.scan_keys import SUMMARIES_KEY

        blocks = blocklist_rag(
            self.blocks,
            ScanOpts(summaries=True, summary_threshold=2),
        )
        print(serialize_blocks(blocks))
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT, [SUMMARIES_KEY]
        )
        print(serialize_chunks(chunks))
        for c in chunks:
            self.assertIn(SUMMARIES_KEY, c.metadata)

    def test_notskipped_havetitles(self):
        """Test all non-skipped chunked blocks have titles"""
        from lmm.scan.scan_keys import TITLES_KEY

        blocks = blocklist_rag(self.blocks, ScanOpts(titles=True))
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT, [TITLES_KEY]
        )
        for c in chunks:
            self.assertIn(TITLES_KEY, c.metadata)


class TestAnnotationModel(unittest.TestCase):
    def test_add_inherited(self):
        am = AnnotationModel()
        am.add_inherited_properties("prop1")
        self.assertIn("prop1", am.inherited_properties)
        am.add_inherited_properties(["prop2", "prop3"])
        self.assertIn("prop2", am.inherited_properties)
        self.assertIn("prop3", am.inherited_properties)

    def test_add_own(self):
        am = AnnotationModel()
        am.add_own_properties("prop1")
        self.assertIn("prop1", am.own_properties)
        am.add_own_properties(["prop2", "prop3"])
        self.assertIn("prop2", am.own_properties)

    def test_has_property(self):
        am = AnnotationModel(inherited_properties=["p1"], own_properties=["p2"])
        self.assertTrue(am.has_property("p1"))
        self.assertTrue(am.has_property("p2"))
        self.assertFalse(am.has_property("p3"))

    def test_has_properties(self):
        am = AnnotationModel()
        self.assertFalse(am.has_properties())
        am.add_inherited_properties("p1")
        self.assertTrue(am.has_properties())


class TestEncodingModelWarnings(unittest.TestCase):
    def test_sparse_warning(self):
        from lmm.utils.logging import LoglistLogger
        import logging

        logger = LoglistLogger()
        blocks_to_chunks(blocks, EncodingModel.SPARSE, logger=logger)
        self.assertTrue(logger.count_logs(level=logging.ERROR) > 0)

    def test_merged_warning(self):
        from lmm.utils.logging import LoglistLogger
        import logging

        logger = LoglistLogger()
        blocks_to_chunks(blocks, EncodingModel.MERGED, logger=logger)
        self.assertTrue(logger.count_logs(level=logging.WARNING) > 0)


if __name__ == '__main__':
    unittest.main()
