"""
Utilities to work with lists of markdown blocks.

Note: Most functions mutate the content of block lists in place. To avoid this
or maintain referential transparency, call `blocklist_copy()` from 
`lmm.markdown.parse_markdown` before using these functions.

Main Functions:
- `compose()`: Compose multiple block processing functions
- `clear_metadata()`: Remove metadata blocks from lists
- `clear_metadata_properties()`: Remove specific properties from metadata blocks
- `merge_textblocks()`: Merge contiguous text blocks
- `unmerge_textblocks()`: Split merged text blocks at blank lines
- `merge_textblocks_if()`: Conditionally merge text blocks based on predicate
- `merge_equation_blocks()`: Merge text blocks separated by equations
- `merge_code_blocks()`: Merge text blocks separated by code blocks
- `merge_short_textblocks()`: Merge short text blocks based on word count

Behaviour:
All functions are pure in the sense that they do not raise exceptions under normal
usage. They accept well-formed block lists and return transformed block lists.
No custom logger is used; functions follow a functional programming style.
"""

# pyright: reportUnusedFunction=false

from collections.abc import Callable
from functools import reduce
import re

from .parse_markdown import (
    Block,
    TextBlock,
    MetadataBlock,
)
from .parse_markdown import serialize_blocks, parse_markdown_text

# type of functions that process block lists
BlockFunc = Callable[[list[Block]], list[Block]]


def compose(*funcs: BlockFunc) -> BlockFunc:
    """
    Compose multiple functions that process lists of Block objects.
    Functions are applied from left to right, so compose(f, g, h)(x)
    is equivalent to h(g(f(x))).

    Args:
        *funcs: Variable number of functions, each taking a
            list[Block] and returning a list[Block]

    Returns:
        A function that applies all input functions in sequence. If no
        functions are provided, returns the identity function. If one
        function is provided, returns that function.
    """
    if not funcs:
        return lambda x: x

    def _compose_two(f: BlockFunc, g: BlockFunc) -> BlockFunc:
        return lambda x: g(f(x))

    return reduce(_compose_two, funcs)


# edit ----------------------------------------------------------


def clear_metadata(blocks: list[Block]) -> list[Block]:
    """
    Remove all metadata blocks from the block list.
    
    Args:
        blocks: List of markdown blocks to filter
        
    Returns:
        New list with all MetadataBlock instances removed
    """
    return [b for b in blocks if b.type != "metadata"]


def clear_metadata_properties(
    blocks: list[Block], keys: list[str]
) -> list[Block]:
    """
    Remove key/value properties from metadata blocks as specified by keys.
    
    Metadata blocks with no remaining properties after removal are deleted unless
    they contain private metadata (private_ field).
    
    Args:
        blocks: List of markdown blocks to process
        keys: List of property keys to remove from metadata blocks
        
    Returns:
        New list with specified properties removed from MetadataBlock instances.
        MetadataBlocks that become empty (no content and no private_ data) are
        excluded from the result.
    """
    if not keys:
        return blocks

    blocklist: list[Block] = []
    for b in blocks:
        if isinstance(b, MetadataBlock):
            newb: MetadataBlock = b.deep_copy()
            for k in keys:
                newb.content.pop(k, None)
            if len(newb.content) > 0 or bool(newb.private_):
                blocklist.append(newb)
        else:
            blocklist.append(b)
    return blocklist


def merge_textblocks(blocks: list[Block]) -> list[Block]:
    """
    Merge contiguous text blocks into larger blocks.
    
    Args:
        blocks: List of markdown blocks to process
        
    Returns:
        New list where consecutive TextBlock instances have been merged using
        serialize_blocks to create combined content.

    Example:
        ```python
        # three blocks
        blocks = [
            HeadingBlock(content = "Title")
            TextBlock(content = "Text 1")
            TextBlock(content = "Text 2")
        ]
        # creates two blocks, heading and text
        newblocks = merge_textblocks(blocks)
        ```
    """
    blocklist: list[Block] = []
    text_stack: list[Block] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            # shift
            text_stack.append(b)
        else:
            if len(text_stack) > 0:
                # we have something else than a text block,
                # reduce existing text blocks...
                blocklist.append(
                    TextBlock(content=serialize_blocks(text_stack))
                )
                text_stack.clear()
            # always reduce other blocks
            blocklist.append(b)

    if len(text_stack) > 0:
        # residual text blocks at end of document
        blocklist.append(
            TextBlock(content=serialize_blocks(text_stack))
        )

    return blocklist


def unmerge_textblocks(blocks: list[Block]) -> list[Block]:
    """
    Unmerge text blocks separated by blank lines. This function is the inverse
    of merge_textblocks.
    
    Args:
        blocks: List of markdown blocks to process
        
    Returns:
        New list where TextBlock instances have been split at blank lines using
        parse_markdown_text.
    """

    blocklist: list[Block] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            blocklist.extend(parse_markdown_text(b.get_content()))
        else:
            blocklist.append(b)
    return blocklist


def merge_textblocks_if(
    blocks: list[Block], test_func: Callable[[TextBlock], bool]
) -> list[Block]:
    """
    Merge text blocks together that are separated by blocks for which
    test_func(block) is true.
    
    Args:
        blocks: List of markdown blocks to process
        test_func: Predicate function that takes a TextBlock and returns True
            if the block should act as a separator triggering merges
            
    Returns:
        New list where TextBlock instances are merged when separated by blocks
        for which test_func returns True.

    Example:
        ```python
        blocks = [
            TextBlock(content = "Text 1")
            TextBlock(content = "Lext 2")
            TextBlock(content = "Text 3")
        ]
        # This creates one single block
        newblocks = merge_textblocks_if(blocks,
            lambda x: x.get_content().startswith("Lext"))

        # These will also be one single block
        newblocks = merge_textblocks_if(blocks[0:1],
            lambda x: x.get_content().startswith("Lext"))
        newblocks = merge_textblocks_if(blocks[1:2],
            lambda x: x.get_content().startswith("Lext"))

        # This leaves blocks unchanged
        newblocks = merge_textblocks_if(blocks,
            lambda x: x.get_content().startswith("Q"))
        ```
    """

    if not blocks:
        return []

    test_func_withnone: Callable[[TextBlock | None], bool] = (
        lambda x: (  # noqa: E731
            test_func(x) if x is not None else False
        )
    )

    blocklist: list[Block] = []
    curblock: Block = blocks[0].deep_copy()
    lastappend: TextBlock | None = None
    if isinstance(curblock, TextBlock) and test_func(curblock):
        lastappend = curblock
    for b in blocks[1:]:
        match b:
            case TextBlock() as bl if test_func(bl):
                if isinstance(curblock, TextBlock):
                    curblock.extend(bl)  # shift
                    lastappend = bl
                else:
                    # reduce
                    blocklist.append(curblock)
                    curblock = bl.deep_copy()
                    lastappend = bl
            case TextBlock() as bl:
                if isinstance(
                    curblock, TextBlock
                ) and test_func_withnone(lastappend):
                    curblock.extend(bl)  # shift
                    lastappend = bl
                else:
                    # reduce
                    blocklist.append(curblock)
                    curblock = bl.deep_copy()
                    lastappend = None
            case _:  # reduce
                blocklist.append(curblock)
                curblock = b
                lastappend = None
    blocklist.append(curblock)  # reduce

    return blocklist


def merge_code_blocks(
    blocks: list[Block], linecount: int = 12
) -> list[Block]:
    """
    Merge text blocks together that are separated by code blocks of size less
    or equal to linecount.
    
    Args:
        blocks: List of markdown blocks to process
        linecount: Maximum number of lines in code blocks that will trigger
            merging (default: 12)
            
    Returns:
        New list where TextBlock instances are merged when separated by small
        code blocks (markdown fenced code blocks with ``` delimiters).
    """

    def _is_code_block(b: TextBlock) -> bool:
        content: str = b.get_content()
        return (
            re.match(
                r"^```(\{[^\n]*\}|(\w+))?\n(.*?)\n```$",
                content,
                re.DOTALL,
            )
            is not None
        ) and (content.count("\n") <= (linecount + 1))

    return merge_textblocks_if(blocks, _is_code_block)


def merge_equation_blocks(blocks: list[Block]) -> list[Block]:
    """
    Merge text blocks together that are separated by equations.
    
    Equations are identified as text blocks matching the pattern $$...$$
    (LaTeX display math delimiters).
    
    Args:
        blocks: List of markdown blocks to process
        
    Returns:
        New list where TextBlock instances are merged when separated by
        equation blocks.
    """

    def _is_eq_block(block: TextBlock) -> bool:
        return (
            re.match(r"^\s*\$\$.*\$\$\s*$", block.get_content())
            is not None
        )

    return merge_textblocks_if(blocks, _is_eq_block)


def _find_largest_divisor(number: int, threshold: int) -> int:
    """
    Finds the largest integer divisor of 'number' such that
    'number' / divisor <= 'threshold'.

    Args:
        number (int): The given number.
        threshold (int): The maximum allowed result of the division.

    Returns:
        int: The largest integer divisor satisfying the condition, or
            1 if the threshold is larger than the number.
    """
    if threshold >= number:
        return 1

    divisors: set[int] = set()
    for i in range(1, int(number**0.5) + 1):
        if number % i == 0:
            divisors.add(i)
            divisors.add(number // i)

    # Iterate through divisors in descending order to find the largest
    # one that satisfies the condition.
    for d in sorted(list(divisors), reverse=True):
        if number / d <= threshold:
            return d

    # This part should theoretically not be reached if
    # threshold < number and number >= 1, because 'number' itself is
    # always a divisor and 'number / number' = 1, which will always
    # be <= threshold if threshold >= 1.
    # If number is 0, this function would need special handling.
    # Assuming positive integers.
    return 1  # Fallback, though should be covered by the loop
    # returning a divisor


def merge_short_textblocks(
    blocks: list[Block], wordthresh: int = 120
) -> list[Block]:
    """
    Merges short text blocks together, defined by a word count threshold.
    
    Text blocks with fewer than wordthresh words are merged with the next
    text block. This continues until a block meets or exceeds the threshold.
    
    Args:
        blocks: List of markdown blocks to process
        wordthresh: Minimum word count threshold for text blocks (default: 120)
        
    Returns:
        New list where short consecutive TextBlock instances have been merged.
    """

    if not blocks:
        return []

    blocklist: list[Block] = []
    curblock: Block = blocks[0].deep_copy()
    for b in blocks[1:]:
        match b:
            case TextBlock() as bl:
                if (
                    isinstance(curblock, TextBlock)
                    and len(curblock.get_content().split())
                    < wordthresh
                ):
                    curblock.extend(bl)
                else:
                    # reduce
                    blocklist.append(curblock)
                    curblock = bl.deep_copy()
            case _ as bl:  # reduce
                blocklist.append(curblock)
                curblock = bl
    blocklist.append(curblock)  # reduce

    return blocklist
