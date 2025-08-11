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
    merge_textblocks,
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
