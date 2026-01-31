
# pyright: basic
# pyright: reportArgumentType=false

import unittest
from typing import List
import re


from lmm.markdown.parse_markdown import (
    Block,
    MetadataBlock,
    HeadingBlock,
    TextBlock,
)
from lmm.markdown.blockutils import (
    compose,
    clear_metadata,
    clear_metadata_properties,
    merge_textblocks,
    unmerge_textblocks,
    merge_textblocks_if,
    merge_equation_blocks,
    merge_short_textblocks,
    merge_code_blocks,
)


class TestCompose(unittest.TestCase):
    """Test the compose function that combines multiple block processing functions."""

    def test_empty_compose(self):
        """Test that compose with no functions returns the identity function."""
        blocks: list[Block] = [TextBlock(content="Test")]
        identity_func = compose()
        self.assertEqual(blocks, identity_func(blocks))

    def test_single_function(self):
        """Test that compose with a single function returns that function."""
        blocks: list[Block] = [
            MetadataBlock(content={"key": "value"}),
            TextBlock(content="Test"),
        ]
        result = compose(clear_metadata)(blocks)
        self.assertEqual(result, clear_metadata(blocks))

    def test_multiple_functions(self):
        """Test that compose with multiple functions applies them in order."""
        # Create test blocks
        blocks: list[Block] = [
            MetadataBlock(content={"key": "value"}),
            TextBlock(content="Block 1"),
            TextBlock(content="Block 2"),
        ]

        # Define a simple test function
        def add_test_block(blocks: List[Block]) -> List[Block]:
            return blocks + [TextBlock(content="Added Block")]

        # Compose functions and apply
        composed = compose(clear_metadata, add_test_block)
        result = composed(blocks)

        # Verify the result
        expected = [
            TextBlock(content="Block 1"),
            TextBlock(content="Block 2"),
            TextBlock(content="Added Block"),
        ]
        self.assertEqual(len(result), len(expected))
        for i, block in enumerate(result):
            self.assertEqual(
                block.get_content(), expected[i].get_content()
            )

    def test_function_order(self):
        """Test that functions in compose are applied from left to right."""
        blocks: list[Block] = [TextBlock(content="Original")]

        def func1(blocks: List[Block]) -> List[Block]:
            return [TextBlock(content="Func1")]

        def func2(blocks: List[Block]) -> List[Block]:
            return [TextBlock(content="Func2")]

        # Order matters: func1 then func2
        result1 = compose(func1, func2)(blocks)
        self.assertEqual(result1[0].get_content(), "Func2")

        # Reverse order: func2 then func1
        result2 = compose(func2, func1)(blocks)
        self.assertEqual(result2[0].get_content(), "Func1")


class TestClearMetadata(unittest.TestCase):
    """Test the clear_metadata function that removes metadata blocks."""

    def test_empty_list(self):
        """Test with an empty list of blocks."""
        self.assertEqual(clear_metadata([]), [])

    def test_no_metadata(self):
        """Test with a list that has no metadata blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="Text 2"),
        ]
        result = clear_metadata(blocks)
        self.assertEqual(result, blocks)

    def test_with_metadata(self):
        """Test with a list that has metadata blocks."""
        blocks: list[Block] = [
            MetadataBlock(content={"key1": "value1"}),
            TextBlock(content="Text 1"),
            MetadataBlock(content={"key2": "value2"}),
            TextBlock(content="Text 2"),
        ]
        result = clear_metadata(blocks)
        expected = [
            TextBlock(content="Text 1"),
            TextBlock(content="Text 2"),
        ]
        self.assertEqual(len(result), len(expected))
        for i, block in enumerate(result):
            self.assertEqual(
                block.get_content(), expected[i].get_content()
            )

    def test_only_metadata(self):
        """Test with a list that has only metadata blocks."""
        blocks: list[Block] = [
            MetadataBlock(content={"key1": "value1"}),
            MetadataBlock(content={"key2": "value2"}),
        ]
        result = clear_metadata(blocks)
        self.assertEqual(result, [])


class TestClearMetadataProperties(unittest.TestCase):
    """Test the clear_metadata function that removes metadata blocks."""

    def test_empty_list(self):
        """Test with an empty list of blocks."""
        self.assertEqual(clear_metadata_properties([], []), [])

    def test_no_metadata(self):
        """Test with a list that has no metadata blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="Text 2"),
        ]
        result = clear_metadata_properties(blocks, ["summary"])
        self.assertEqual(result, blocks)

    def test_with_metadata(self):
        """Test with a list that has metadata blocks."""
        blocks: list[Block] = [
            MetadataBlock(
                content={"key1": "value1", "key2": "value2"}
            ),
            TextBlock(content="Text 1"),
            MetadataBlock(content={"key2": "value2"}),
            TextBlock(content="Text 2"),
        ]
        result = clear_metadata_properties(blocks, ["key2"])
        expected = [
            MetadataBlock(content={"key1": "value1"}),
            TextBlock(content="Text 1"),
            TextBlock(content="Text 2"),
        ]
        self.assertEqual(len(result), len(expected))
        for i, block in enumerate(result):
            self.assertEqual(
                block.get_content(), expected[i].get_content()
            )

    def test_only_metadata(self):
        """Test with a list that has only metadata blocks."""
        blocks: list[Block] = [
            MetadataBlock(content={"key1": "value1"}),
            MetadataBlock(content={"key2": "value2"}),
        ]
        result = clear_metadata_properties(blocks, ["key1", "key2"])
        self.assertEqual(len(result), 0)

    def test_only_metadata_several(self):
        """Test with a list that has only metadata blocks."""
        blocks: list[Block] = [
            MetadataBlock(
                content={"key1": "value1", "key3": "value3"}
            ),
            MetadataBlock(
                content={"key2": "value2", "key3": "value3"}
            ),
        ]
        result = clear_metadata_properties(blocks, ["key1", "key2"])
        expected = [
            MetadataBlock(content={"key3": "value3"}),
            MetadataBlock(content={"key3": "value3"}),
        ]
        self.assertEqual(len(result), len(expected))
        for i, block in enumerate(result):
            self.assertEqual(
                block.get_content(), expected[i].get_content()
            )


class TestMergeTextblocks(unittest.TestCase):
    """Test the merge_textblocks function that merges contiguous text blocks."""

    def test_empty_list(self):
        """Test with an empty list of blocks."""
        self.assertEqual(merge_short_textblocks([]), [])
        self.assertEqual(merge_textblocks([]), [])
        self.assertEqual(
            merge_textblocks_if(
                [], lambda x: x.get_content().startswith("Text")
            ),
            [],
        )

    def test_no_text_blocks(self):
        """Test with a list that has no text blocks."""
        blocks: list[Block] = [
            MetadataBlock(content={"key": "value"}),
            HeadingBlock(level=1, content="Heading"),
            HeadingBlock(level=2, content="Heading"),
        ]
        result = merge_short_textblocks(blocks)
        self.assertEqual(result, blocks)
        result = merge_textblocks(blocks)
        self.assertEqual(result, blocks)
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Head")
        )
        self.assertEqual(result, blocks)

    def test_single_text_block(self):
        """Test with a list that has a single text block."""
        blocks: list[Block] = [TextBlock(content="Single text block")]
        result = merge_short_textblocks(blocks)
        # The function adds a newline to the content
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, "text")
        self.assertEqual(result[0].get_content(), "Single text block")
        result = merge_textblocks(blocks)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, "text")
        self.assertEqual(result[0].get_content(), "Single text block")

    def test_contiguous_text_blocks(self):
        """Test with a list that has contiguous text blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(content="Text 2"),
            TextBlock(content="Lext 3"),
            TextBlock(content="Lext 4"),
        ]
        result = merge_short_textblocks(blocks)
        self.assertEqual(blocks[0].get_content(), "Text 1")
        self.assertEqual(blocks[1].get_content(), "Text 2")
        self.assertEqual(blocks[2].get_content(), "Lext 3")
        self.assertEqual(blocks[3].get_content(), "Lext 4")
        self.assertEqual(len(result), 1)
        # TextBlock.append adds a newline between blocks
        self.assertEqual(
            result[0].get_content(),
            "Text 1\n\nText 2\n\nLext 3\n\nLext 4",
        )
        result = merge_textblocks(blocks)
        self.assertEqual(len(result), 1)
        # TextBlock.append adds a newline between blocks
        self.assertEqual(
            result[0].get_content(),
            "Text 1\n\nText 2\n\nLext 3\n\nLext 4",
        )
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Lext")
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].get_content(), "Text 1")
        self.assertEqual(
            result[1].get_content(), "Text 2\n\nLext 3\n\nLext 4"
        )
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Text")
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0].get_content(), "Text 1\n\nText 2\n\nLext 3"
        )
        self.assertEqual(result[1].get_content(), "Lext 4")
        result = merge_textblocks_if(blocks, lambda _: True)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].get_content(),
            "Text 1\n\nText 2\n\nLext 3\n\nLext 4",
        )

    def test_contiguous_text_blocks2(self):
        """Test with a list that has contiguous text blocks, but
        a limit of 3 words"""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(content="Lext 2"),
            TextBlock(content="Text 3"),
            TextBlock(content="Text 4"),
        ]
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Text")
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].get_content(),
            "Text 1\n\nLext 2\n\nText 3\n\nText 4",
        )
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Lext")
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0].get_content(),
            "Text 1\n\nLext 2\n\nText 3",
        )
        self.assertEqual(
            result[1].get_content(),
            "Text 4",
        )
        result = merge_short_textblocks(blocks, 3)
        self.assertEqual(len(result), 2)
        result = merge_textblocks(blocks)
        self.assertEqual(len(result), 1)

    def test_contiguous_text_blocks3(self):
        """Test with a list that has contiguous text blocks, and
        terminates with a heading block"""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(content="Text 2"),
            TextBlock(content="Text 3"),
            HeadingBlock(level=3, content="Heading content"),
        ]
        result = merge_short_textblocks(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0].get_content(), "Text 1\n\nText 2\n\nText 3"
        )
        self.assertEqual(blocks[-1].get_content(), "Heading content")
        result = merge_textblocks(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0].get_content(), "Text 1\n\nText 2\n\nText 3"
        )
        self.assertEqual(blocks[-1].get_content(), "Heading content")
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Text")
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0].get_content(), "Text 1\n\nText 2\n\nText 3"
        )
        self.assertEqual(blocks[-1].get_content(), "Heading content")
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Lext")
        )
        self.assertEqual(len(result), len(blocks))

    def test_contiguous_text_blocks4(self):
        """Test with a list that has contiguous text blocks, and
        terminates with a header block"""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(content="Text 2"),
            TextBlock(content="Lext 3"),
            MetadataBlock(content={'content': "Heading content"}),
        ]
        result = merge_short_textblocks(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0].get_content(), "Text 1\n\nText 2\n\nLext 3"
        )
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Lext")
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Text 1")
        self.assertEqual(result[1].get_content(), "Text 2\n\nLext 3")
        result = merge_textblocks(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0].get_content(), "Text 1\n\nText 2\n\nLext 3"
        )

    def test_mixed_blocks(self):
        """Test with a list that has mixed block types."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(content="Text 2"),
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="Text 3"),
            TextBlock(content="Text 4"),
        ]
        result = merge_short_textblocks(blocks)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Text 1\n\nText 2")
        self.assertEqual(result[1].get_content(), "Heading")
        self.assertEqual(result[2].get_content(), "Text 3\n\nText 4")
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Text")
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Text 1\n\nText 2")
        self.assertEqual(result[1].get_content(), "Heading")
        self.assertEqual(result[2].get_content(), "Text 3\n\nText 4")
        result = merge_textblocks(blocks)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Text 1\n\nText 2")
        self.assertEqual(result[1].get_content(), "Heading")
        self.assertEqual(result[2].get_content(), "Text 3\n\nText 4")

    def test_mixed_blocks2(self):
        """Test with a list that has mixed block types."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(content="Text 2"),
            MetadataBlock(content={'content': "Heading"}),
            TextBlock(content="Text 3"),
            TextBlock(content="Text 4"),
        ]
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Text")
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Text 1\n\nText 2")
        self.assertDictEqual(
            result[1].get_content(), {'content': "Heading"}
        )
        self.assertEqual(result[2].get_content(), "Text 3\n\nText 4")
        result = merge_short_textblocks(blocks)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Text 1\n\nText 2")
        self.assertDictEqual(
            result[1].get_content(), {'content': "Heading"}
        )
        self.assertEqual(result[2].get_content(), "Text 3\n\nText 4")
        result = merge_textblocks(blocks)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Text 1\n\nText 2")
        self.assertDictEqual(
            result[1].get_content(), {'content': "Heading"}
        )
        self.assertEqual(result[2].get_content(), "Text 3\n\nText 4")

    def test_text_blocks_at_end(self):
        """Test with text blocks at the end of the list."""
        blocks: list[Block] = [
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="Text 1"),
            TextBlock(content="Text 2"),
        ]
        result = merge_textblocks_if(
            blocks, lambda x: x.get_content().startswith("Qext")
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Heading")
        self.assertEqual(result[1].get_content(), "Text 1")
        self.assertEqual(result[2].get_content(), "Text 2")
        result = merge_textblocks(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].get_content(), "Heading")
        self.assertEqual(result[1].get_content(), "Text 1\n\nText 2")
        result = merge_short_textblocks(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].get_content(), "Heading")
        self.assertEqual(result[1].get_content(), "Text 1\n\nText 2")


class TestPoolEquationBlocks(unittest.TestCase):
    """Test the merge_equation_blocks function that pools text blocks separated by equations."""

    def test_empty_list(self):
        """Test with an empty list of blocks."""
        self.assertEqual(merge_equation_blocks([]), [])

    def test_no_equation_blocks(self):
        """Test with a list that has no equation blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="Text 2"),
        ]
        result = merge_equation_blocks(blocks)
        self.assertEqual(len(result), len(blocks))
        for i, block in enumerate(result):
            self.assertEqual(
                block.get_content(), blocks[i].get_content()
            )

    def test_with_equation_blocks(self):
        """Test with a list that has equation blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text before equation"),
            TextBlock(content="$$E = mc^2$$"),
            TextBlock(content="Text after equation"),
        ]
        result = merge_equation_blocks(blocks)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].get_content(),
            "Text before equation\n\n$$E = mc^2$$\n\nText after equation",
        )

    def test_with_equation_blocks2(self):
        """Test with a list that has equation blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text before equation"),
            TextBlock(content="$$E = mc^2$$   "),
            TextBlock(content="Text after equation"),
        ]
        result = merge_equation_blocks(blocks)
        self.assertEqual(len(result), 1)

    def test_with_equation_blocks3(self):
        """Test with a list that has equation blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text before equation"),
            TextBlock(content="     $$E = mc^2$$"),
            TextBlock(content="Text after equation"),
        ]
        result = merge_equation_blocks(blocks)
        self.assertEqual(len(result), 1)

    def test_multiple_equation_blocks(self):
        """Test with multiple equation blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(content="$$E = mc^2$$"),
            TextBlock(content="Text 2"),
            TextBlock(content="$$F = ma$$"),
            TextBlock(content="Text 3"),
            TextBlock(content="Text 4"),
        ]
        result = merge_equation_blocks(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].get_content(), "Text 4")

    def test_multiple_equation_blocks2(self):
        """Test with multiple equation blocks."""
        blocks: list[Block] = [
            TextBlock(content="$$E = mc^2$$"),
            TextBlock(content="$$F = ma$$"),
            TextBlock(content="Text 3"),
            TextBlock(content="Text 4"),
        ]
        result = merge_equation_blocks(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].get_content(), "Text 4")

    def test_multiple_equation_blocks3(self):
        """Test with multiple equation blocks."""
        blocks: list[Block] = [
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="$$E = mc^2$$"),
            TextBlock(content="$$F = ma$$"),
            HeadingBlock(level=1, content="Heading"),
        ]
        result = merge_equation_blocks(blocks)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Heading")
        self.assertEqual(result[2].get_content(), "Heading")

    def test_equation_at_start(self):
        """Test with an equation block at the start."""
        blocks: list[Block] = [
            TextBlock(content="$$E = mc^2$$"),
            TextBlock(content="Text after equation"),
        ]
        result = merge_equation_blocks(blocks)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].get_content(),
            "$$E = mc^2$$\n\nText after equation",
        )

    def test_equation_at_end(self):
        """Test with an equation block at the end."""
        blocks: list[Block] = [
            TextBlock(content="Text before equation"),
            TextBlock(content="$$E = mc^2$$"),
        ]
        result = merge_equation_blocks(blocks)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].get_content(),
            "Text before equation\n\n$$E = mc^2$$",
        )

    def test_with_non_text_blocks(self):
        """Test with non-text blocks between equation blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(content="$$E = mc^2$$"),
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="$$F = ma$$"),
            TextBlock(content="Text 2"),
        ]
        result = merge_equation_blocks(blocks)
        # self.assertEqual(len(result), 3)
        self.assertEqual(
            result[0].get_content(), "Text 1\n\n$$E = mc^2$$"
        )
        self.assertEqual(result[1].get_content(), "Heading")
        self.assertEqual(
            result[2].get_content(), "$$F = ma$$\n\nText 2"
        )


class TestPoolCodeBlocks(unittest.TestCase):
    """Test the merge_code_blocks function that pools text blocks separated by code blocks."""

    def test_code_regexp(self):
        def _is_code_block(b: TextBlock) -> bool:
            content: str = b.get_content()
            return (
                re.match(
                    r"^```(\{[^\n]*\}|(\w+))?\n(.*?)\n```$",
                    content,
                    re.DOTALL,
                )
                is not None
            )

        def txtbl(x: str) -> TextBlock:
            return TextBlock.from_text(x)

        """Test the regular expression for code"""
        self.assertTrue(
            _is_code_block(txtbl("```{r}\nfit <- lm(y ~ x)\n```"))
        )
        self.assertTrue(
            _is_code_block(
                txtbl("```\nfit <- lm(y ~ x)\nsummary(fit)\n```")
            )
        )
        self.assertTrue(
            _is_code_block(
                txtbl("```{r, echo = T}\nfit <- lm(y ~ x)\n```")
            )
        )
        self.assertTrue(
            _is_code_block(
                txtbl("```{r test, echo = F}\nfit <- lm(y ~ x)\n```")
            )
        )
        self.assertTrue(
            _is_code_block(
                txtbl(
                    """```
                                            fit <- lm(y ~ x, data = data)\n```"""
                )
            )
        )
        self.assertFalse(
            _is_code_block(txtbl("```{r\nfit <- lm(y ~ x)\n```"))
        )
        self.assertFalse(
            _is_code_block(txtbl("```{r}\nfit <- lm(y ~ x)\n``"))
        )

    def test_empty_list(self):
        """Test with an empty list of blocks."""
        self.assertEqual(merge_code_blocks([]), [])

    def test_no_code_blocks(self):
        """Test with a list that has no code blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="Text 2"),
        ]
        result = merge_code_blocks(blocks)
        self.assertEqual(len(result), len(blocks))
        for i, block in enumerate(result):
            self.assertEqual(
                block.get_content(), blocks[i].get_content()
            )

    def test_with_code_blocks(self):
        """Test with a list that has code blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text before code"),
            TextBlock(
                content="```{r}\nfit ~ lm(y ~ x, data = data)\n```"
            ),
            TextBlock(content="Text after code"),
        ]
        result = merge_code_blocks(blocks)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].get_content(),
            "Text before code\n\n```{r}\nfit ~ lm(y ~ x, data = data)\n```\n\nText after code",
        )

    def test_with_multilinecode_blocks(self):
        """Test with a list that has just max line code blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text before code"),
            TextBlock(
                content="""```{r}
                      fit ~ lm(y ~ x, data = data)
                      summary(fit)
                      fit ~ lm(y ~ x, data = data)
                      summary(fit)\n```"""
            ),
            TextBlock(content="Text after code"),
        ]
        result = merge_code_blocks(blocks, 4)
        self.assertEqual(len(result), 1)

    def test_with_longcode_blocks(self):
        """Test with a list that has overthreshold line code blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text before code"),
            TextBlock(
                content="""```{r}
                      fit ~ lm(y ~ x, data = data)
                      summary(fit)
                      fit ~ lm(y ~ x, data = data)
                      summary(fit)
                      fit ~ lm(y ~ x, data = data)
                      summary(fit)
                      fit ~ lm(y ~ x, data = data)
                      summary(fit)
                      fit ~ lm(y ~ x, data = data)
                      summary(fit)
                      fit ~ lm(y ~ x, data = data)
                      summary(fit)\n```"""
            ),
            TextBlock(content="Text after code"),
        ]
        result = merge_code_blocks(blocks, 11)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Text before code")
        self.assertEqual(result[2].get_content(), "Text after code")

    def test_multiple_code_blocks(self):
        """Test with multiple code blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(content="Text 2"),
            TextBlock(
                content="```{r echo = F}\nfit ~ lm(y ~ x, data = data)\n```"
            ),
            TextBlock(content="Text 3"),
            TextBlock(
                content="```\nfit ~ lm(y ~ x, data = data)\n```"
            ),
            TextBlock(content="Text 4"),
            TextBlock(content="Text 5"),
        ]
        result = merge_code_blocks(blocks)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "Text 1")
        self.assertEqual(result[-1].get_content(), "Text 5")

    def test_code_at_start(self):
        """Test with a code block at the start."""
        blocks: list[Block] = [
            TextBlock(
                content="```\nfit ~ lm(y ~ x, data = data)\nsummary(fit)\n```"
            ),
            TextBlock(content="Text after code"),
        ]
        result = merge_code_blocks(blocks)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].get_content(),
            "```\nfit ~ lm(y ~ x, data = data)\nsummary(fit)\n```\n\nText after code",
        )

    def test_code_at_end(self):
        """Test with a code block at the end."""
        blocks: list[Block] = [
            TextBlock(content="Text before code"),
            TextBlock(
                content="```\nfit ~ lm(y ~ x, data = data)\nsummary(fit)\n```"
            ),
        ]
        result = merge_code_blocks(blocks)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result[0].get_content(),
            "Text before code\n\n```\nfit ~ lm(y ~ x, data = data)\nsummary(fit)\n```",
        )

    def test_with_non_text_blocks(self):
        """Test with non-text blocks between code blocks."""
        blocks: list[Block] = [
            TextBlock(content="Text 1"),
            TextBlock(
                content="```\nfit ~ lm(y ~ x, data = data)\n```"
            ),
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="```\nsummary(fit)\n```"),
            TextBlock(content="Text 2"),
        ]
        result = merge_code_blocks(blocks)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            result[0].get_content(),
            "Text 1\n\n```\nfit ~ lm(y ~ x, data = data)\n```",
        )
        self.assertEqual(result[1].get_content(), "Heading")
        self.assertEqual(
            result[2].get_content(),
            "```\nsummary(fit)\n```\n\nText 2",
        )


from lmm.markdown.parse_markdown import (
    parse_markdown_text,
    HeaderBlock,
)

header_block = """
---
author: Roberto Viviani
date: '2025-05-04'
output:
  html_document: default
  md_document: default
  word_document: default
title: Chapter 1
docid: Chapter 1
---

"""

first_heeading_group = """
## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used in practice. Here, we will look at linear models from a practical perspective, emphasizing the issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the *predictors* (also known as *independent variables*), and an *outcome* variable (or *dependent variable*). In the simplest setting, this association may be approximated and displayed by a line relating a predictor and the outcome. However, the generalizations of linear model allow capturing much more complex associations, as we shall see.

There are two broad ways of using linear models. In the first, which is perhaps the most common, we are interested in assessing the relationship between predictors and the outcome in order to establish if this relationship is "significant".

The second use of linear models is to predict the outcome given certain values of the predictors.[^1] This use of linear models is the same as in machine learning.[^2] In this case, after the fit of a linear model has been computed, one may use values of the predictors that the model had not seen to predict the outcome.

[^1]: Predictor means the same as independent variable.

[^2]: Machine learning is the field of artificial intelligence that is concerned with training programs to accomplish tasks based on a training set.

"""
second_heading_group = """
---
note: revise if possible
...
## Observational and experimental studies

We will first discuss issues arising from assessing the significance of associations.

When we look at the significance of the association between predictors and outcome, it is important to distinguish between two different settings. In the first, we have variables that we have observed in the field. For example, early traumas may exert an influence on the predisposition to psychopathology in adult age. We do not have any way to change the occurrence of traumas in the past and we therefore look at their consequences in a sample from the population. Studies of this type are called *observational*. An important issue in observational studies is that predictors may be *confounded* by other factors affecting the outcome. For example, it is conceivable that traumas may occur more frequently in adverse socioeconomic conditions, and these latter may in turn adversely affect the predisposition to psychopathology. When we assess the association of traumas and psychopathology, it may include the effects of socioeconomic conditions, as traumatized individuals are also those that are most disadvantaged socioeconomically.

In the second setting, the predictor of interest is a variable representing an experimental manipulation, such as treatment, changes in aspects of a cognitive task, etc. A key aspect of *experimental* studies is that the value of the predictor of interest, being determined by you, the experimenter, can be randomized. For this reason, at least in the long term, the action of other possible variables on the outcome cancels out, as there is no relationship between these potential confounders and the predictor of interest. We cannot randomize traumas, but if we could, we would no longer have the potential confounder of adverse socioeconomic conditions, because randomization ensures that, in the long run, individuals from all conditions are equally traumatized.

Importantly, the way in which we estimate models in observational and experimental studies is exactly the same. However, the conclusions that we may draw from establishing the likely existence of an association between predictor and outcome differ in these two cases. In observational studies, we can only infer the mere association, unless a sometimes fairly extensive efforts are made on being able to contain the effect of confounders. In experimental studies, we can infer that the treatment variable causes the effect measured by the outcome variable. The term *effect* is sometimes reserved to the causal association that may be established in experimental studies, but is often used more loosely to mean any association between predictors and outcomes. This loose usage may mask the very different nature of the associations detected in these two types of studies.

### Using models for prediction

When using linear models solely for prediction, the concerns regarding inference in observational studies do not apply. While confounding in predictor variables can influence the interpretation of the model coefficients, it may not significantly affect predictive performance. Therefore, if our primary focus is on achieving accurate predictions of the outcome, we can simply include all predictors in the model that contribute to reduce the prediction error.
"""

composite_text_group = """
---
summary: This section introduces the model equation, emphasizing its importance in understanding the linear model.
...
## The model equation

It is now time to see a linear model in action. Let us assume we measured depressive symptoms (the outcome) with a self-rating scale and we are interested in exploring their association with sex. We may formulate the model as follows:

depression = baseline + female + error

Here, $female$ encodes sex by taking the value 1 for women and 0 for men. The outcome we measured (depressive symptoms) is decomposed by the model into the sum of three terms: the baseline depressiveness, i.e. the average depressive symptoms of males; the difference in depressive symptoms observed on average in females relative to males, and a final term accounting for the difference in depressive symptoms measured in each individuals relative to these averages. This expression applies for each observation. If we index an observation with $i$, we may write

$$depression_i = baseline + female_i * female_coefficient + error_i$$

This equation is called the _model equation_. At this point, you might be tempted to skip over the model equation and think that you may deal with it later or avoid using equations altogether. Don't! The model equation is crucial to understand linear models. Once mastered, it becomes your most helpful aid in understanding the model. It is therefore crucial to become familiar with it. 
This model equation means that in individual $i$, the measured depressive symptoms $depression_i$ are the sum of  $baseline$, which is the average depressive symptoms in male individuals, of the average difference in depressive symptoms in females $female_coefficient$, and of the errors $error_i$. The $female_coefficient$ is applied to all all data, but here in males it has no influence as it is multiplied by $female$, which is zero in males. In females, the $female_coefficient$ is multiplied by one and remains equal to itself. The errors are the difference between the observed depressiveness in each individual and these estimated averages. Note that there is one baseline and one female coefficient for the whole dataset. These are the _coefficients_ of the model. The baseline coefficient is sometimes called _constant term_ or _intercept_.

When the model is _fitted_, we obtain the estimates of the coefficients. To do this, we use the following line in the R console:

```{r}
fit <- lm(depression ~ female, data = depr_sample)
```

where depr_sample is the data frame containing the columns `depression` and `female`. The expression `depression ~ female` is the model equation, except for the baseline term, which is added automatically by R. We could also have written this term explicitly in the model equation as follows,

```{r}
fit <- lm(depression ~ 1 + female, data = depr_sample)
```

obtaining the same model. We can now inspect the coefficients of the model fit:

```{r}
summary(fit)
```

To understand what these coefficients mean, let us put them in the model equation. We obtain

$$depression_i = 24 + female_i * 4 + error_i$$

To calculate the predicted depressiveness, we set the value of the female predictor to one or zero, depending on the sex of the individual we are predicting. It follows that 24 is the predicted depressive symptom score in males, and 4 is the predicted difference in these scores from males to females. It also follows that the predicted depression score in females is 24 + 4 = 28 (in many samples of self-reported depressiveness, one often finds higher scores in females, although this finding may fail to be confirmed by other measures of depressiveness).

While this is a linear model, it is also a two-sample _t_ test. In a two-sample _t_ test we test the difference between means. Here, these means are 24 and 28. The linear model is a very general formulation, relative to which other models are special cases. 

Another case of a linear model that you might not have suspected to be one is the mean:

$$depression_i = constant_term + error_i$$

where we used the name _constant term_ for the baseline, estimated as the average of all observed depression scores in the whole sample (which will differ from the estimate of the male average of the previous model). This is the simplest possible linear model, and we will come back to it later to help intuition about the properties of the fitted model.
"""

document = (
    header_block
    + first_heeading_group
    + second_heading_group
    + composite_text_group
)


class TestPooling(unittest.TestCase):
    # Futher pooling tests

    def test_pooling(self):
        # check pooling all text under headings
        blocks: list[Block] = parse_markdown_text(document)
        count_heads = len(
            [
                b
                for b in blocks
                if isinstance(b, HeadingBlock)
                and not isinstance(b, HeaderBlock)
            ]
        )

        blocks = merge_textblocks(blocks)
        self.assertEqual(
            count_heads,
            len([b for b in blocks if isinstance(b, TextBlock)]),
        )

    def test_pooling_eq_code(self):
        new_doc = (
            header_block
            + """

# Text with complex stuff

This introduces an equation:

$$y = x + 1$$

and this a code block:

```{r}
fit ~ lm(y ~x)
```

and final text.
"""
        )
        blocks: list[Block] = parse_markdown_text(new_doc)

        preproc_chain = compose(
            merge_equation_blocks,
            merge_code_blocks,
        )

        blocks = preproc_chain(blocks)
        self.assertEqual(
            1, len([b for b in blocks if isinstance(b, TextBlock)])
        )

    def test_pooling_eq_code_python(self):
        new_doc = (
            header_block
            + """

# Text with complex stuff

This introduces an equation:

$$y = x + 1$$

and this a code block:

```{python}
fit ~ lm(y ~x)
```

and final text.
"""
        )
        blocks: list[Block] = parse_markdown_text(new_doc)

        preproc_chain = compose(
            merge_equation_blocks,
            merge_code_blocks,
        )

        blocks = preproc_chain(blocks)
        self.assertEqual(
            1, len([b for b in blocks if isinstance(b, TextBlock)])
        )

    def test_pooling_eq_code2(self):
        new_doc = (
            header_block
            + """

# Text with complex stuff

This introduces an equation:

$$y = x + 1$$

and this a code block:

```
fit ~ lm(y ~x)
```

and final text.
"""
        )
        blocks: list[Block] = parse_markdown_text(new_doc)

        preproc_chain = compose(
            merge_equation_blocks,
            merge_code_blocks,
        )

        blocks = preproc_chain(blocks)
        self.assertEqual(
            1, len([b for b in blocks if isinstance(b, TextBlock)])
        )

    def test_pooling_eq_code3(self):
        new_doc = (
            header_block
            + """

# Text with complex stuff

This introduces an equation:

$$y = x + 1$$

and these are code blocks:

```python
fit ~ lm(y ~x)
```


```r
fit ~ lm(y ~x)
```

and final text.
"""
        )
        blocks: list[Block] = parse_markdown_text(new_doc)

        preproc_chain = compose(
            merge_equation_blocks,
            merge_code_blocks,
        )

        blocks = preproc_chain(blocks)
        self.assertEqual(
            1, len([b for b in blocks if isinstance(b, TextBlock)])
        )

    def test_unmerge_textblocks(self):
        text = (
            "First paragraph\n\nSecond paragraph\n\nThird paragraph"
        )
        blocklist = parse_markdown_text(text)
        self.assertEqual(len(blocklist), 3)
        blocks = merge_textblocks(blocklist)
        self.assertEqual(len(blocks), 1)
        blocks = unmerge_textblocks(blocks)
        self.assertEqual(len(blocks), 3)
        for b, c in zip(blocklist, blocks):
            self.assertEqual(b.get_content(), c.get_content())


class TestPoolShortTextblocks(unittest.TestCase):
    """Test the merge_short_textblocks function that merges short text blocks."""

    def test_empty_list(self):
        """Test with an empty list of blocks."""
        self.assertEqual(merge_short_textblocks([]), [])

    def test_no_text_blocks(self):
        """Test with a list that has no text blocks."""
        blocks: list[Block] = [
            MetadataBlock(content={"key": "value"}),
            HeadingBlock(level=1, content="Heading"),
        ]
        result = merge_short_textblocks(blocks)
        self.assertEqual(result, blocks)

    def test_single_text_block(self):
        """Test with a list that has a single text block."""
        blocks: list[Block] = [TextBlock(content="Single text block")]
        result = merge_short_textblocks(blocks)
        self.assertEqual(result, blocks)

    def test_short_text_blocks(self):
        """Test with a list that has short text blocks."""
        blocks: list[Block] = [
            TextBlock(content="Short text 1"),
            TextBlock(content="Short text 2"),
            TextBlock(content="Short text 3"),
        ]
        result = merge_short_textblocks(blocks, wordthresh=5)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0].get_content(), "Short text 1\n\nShort text 2"
        )
        self.assertEqual(result[1].get_content(), "Short text 3")

    def test_mixed_length_blocks(self):
        """Test with a list that has mixed length text blocks."""
        long_text = "This is a long text block with more than 5 words so it should not be merged with the previous block"
        blocks: list[Block] = [
            TextBlock(content="Short text 1"),
            TextBlock(content="Short text 2"),
            TextBlock(content=long_text),
            TextBlock(content="Short text 3"),
        ]
        result = merge_short_textblocks(blocks, wordthresh=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            result[0].get_content(), "Short text 1\n\nShort text 2"
        )
        self.assertEqual(result[1].get_content(), long_text)
        self.assertEqual(result[2].get_content(), "Short text 3")

    def test_with_non_text_blocks(self):
        """Test with non-text blocks between short text blocks."""
        blocks: list[Block] = [
            TextBlock(content="Short text 1"),
            TextBlock(content="Short text 2"),
            HeadingBlock(level=1, content="Heading"),
            TextBlock(content="Short text 3"),
            TextBlock(content="Short text 4"),
        ]
        result = merge_short_textblocks(blocks, wordthresh=5)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            result[0].get_content(), "Short text 1\n\nShort text 2"
        )
        self.assertEqual(result[1].get_content(), "Heading")
        self.assertEqual(
            result[2].get_content(), "Short text 3\n\nShort text 4"
        )

    def test_custom_wordcount(self):
        """Test with a custom wordcount threshold."""
        blocks: list[Block] = [
            TextBlock(content="One two three"),
            TextBlock(content="Four five six seven"),
            TextBlock(content="Eight nine ten eleven twelve"),
        ]
        # With wordcount=3, only the first block is considered short
        result = merge_short_textblocks(blocks, wordthresh=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].get_content(), "One two three")
        self.assertEqual(
            result[1].get_content(), "Four five six seven"
        )
        self.assertEqual(
            result[2].get_content(), "Eight nine ten eleven twelve"
        )

        # With wordcount=4, the first and second blocks are considered short
        result = merge_short_textblocks(blocks, wordthresh=4)
        self.assertEqual(len(result), 2)
        self.assertEqual(
            result[0].get_content(),
            "One two three\n\nFour five six seven",
        )
        self.assertEqual(
            result[1].get_content(), "Eight nine ten eleven twelve"
        )


if __name__ == '__main__':
    unittest.main()
