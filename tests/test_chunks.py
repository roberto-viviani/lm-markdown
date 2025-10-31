"""tests for chunk.py"""

# pyright: basic
# pyright: reportMissingTypeStubs=false

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
)
from lmm.scan.scan_keys import (
    QUESTIONS_KEY,
    CHAT_KEY,
    SUMMARY_KEY,
    TITLES_KEY,
)

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

        new_blocks = blocklist_copy(blocks)
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
        from lmm.scan.scan_rag import scan_rag, ScanOpts
        from lmm.markdown.parse_markdown import blocklist_copy

        self.assertEqual(len(blocks), lenblocks)
        blocklist = scan_rag(
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
            chunk.annotations + ": " + text.get_content(),
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
            chunk.annotations + ": " + text.get_content(),
        )


from lmm.markdown.parse_markdown import parse_markdown_text
from lmm.scan.scan_rag import scan_rag, ScanOpts
from lmm.config.config import Settings, export_settings


class TestSkips(unittest.TestCase):
    # setup and teardown replace config.toml to avoid
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

---
skip: True
...
This is a skipped text block.

# Heading 4 (Process)

Final text block.
"""
        self.blocks = parse_markdown_text(self.doc_with_skip)

    def test_skip_blocks(self):
        """Test that only non-skipped blocks are chunked."""
        blocks = scan_rag(self.blocks)
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)

        # only 2 blocks left
        self.assertEqual(len(chunks), 2)

    def test_notskipped_haveid(self):
        """Test all chunked blocks have id"""
        from lmm.scan.scan_keys import TEXTID_KEY

        blocks = scan_rag(self.blocks)
        chunks = blocks_to_chunks(blocks, EncodingModel.CONTENT)
        for c in chunks:
            self.assertIn(TEXTID_KEY, c.metadata)

    def test_notskipped_havequestions(self):
        """Test all chunked blocks have id"""
        from lmm.scan.scan_keys import QUESTIONS_KEY

        blocks = scan_rag(
            self.blocks,
            ScanOpts(questions=True, questions_threshold=0),
        )
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT, [QUESTIONS_KEY]
        )
        for c in chunks:
            self.assertIn(QUESTIONS_KEY, c.metadata)

    def test_notskipped_havesummaries(self):
        """Test all chunked blocks have id"""
        from lmm.scan.scan_keys import SUMMARIES_KEY

        blocks = scan_rag(
            self.blocks,
            ScanOpts(summaries=True, summary_threshold=0),
        )
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT, [SUMMARIES_KEY]
        )
        for c in chunks:
            self.assertIn(SUMMARIES_KEY, c.metadata)

    def test_notskipped_havetitles(self):
        """Test all chunked blocks have id"""
        from lmm.scan.scan_keys import TITLES_KEY

        blocks = scan_rag(self.blocks, ScanOpts(titles=True))
        chunks = blocks_to_chunks(
            blocks, EncodingModel.SPARSE_CONTENT, [TITLES_KEY]
        )
        for c in chunks:
            self.assertIn(TITLES_KEY, c.metadata)


if __name__ == '__main__':
    unittest.main()
