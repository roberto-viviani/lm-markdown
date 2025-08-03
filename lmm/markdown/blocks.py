"""
Utilities to work with lists of markdown blocks.
Note: call blocklist_copy before using these functions
    to maintain referential transparency.
"""

from functools import reduce
from typing import Callable
import re

from .parse_markdown import (
    Block,
    TextBlock,
)
from .parse_markdown import serialize_blocks
from .tree import HeadingNode, TextNode

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

# map-------------------------------------------------------------


def blocks_map(
    blocks: list[Block],
    map_func: Callable[[Block], Block],
    filter_func: Callable[[Block], bool] = lambda _: True,
) -> list[Block]:
    """Apply map_func to all blocks that satisfy the predicate 
    filter_func"""
    return [map_func(b.deep_copy()) for b in blocks if filter_func(b)]

# edit ----------------------------------------------------------

def clear_metadata(blocks: list[Block]) -> list[Block]:
    """
    Remove metadata blocks from the block list, except the header
    """
    return [b for b in blocks if b.type != "metadata"]


def merge_textblocks(blocks: list[Block]) -> list[Block]:
    """
    Merge contiguous text blocks into larger blocks
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


def pool_textblocks_if(
    blocks: list[Block], test_func: Callable[[TextBlock], bool]
) -> list[Block]:
    """Pool text blocks together that are separated by blocks
    for which test_func(block) is true"""

    if not blocks:
        return []

    test_func_withnone: Callable[[TextBlock | None], bool] = (
        lambda x: test_func(x) if x is not None else False
    )

    blocklist: list[Block] = []
    curblock: Block = blocks[0]
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
                    curblock = bl
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
                    curblock = bl
                    lastappend = None
            case _ as bl:  # reduce
                blocklist.append(curblock)
                curblock = bl
                lastappend = None
    blocklist.append(curblock)  # reduce

    return blocklist


def pool_code_blocks(
    blocks: list[Block], linecount: int = 12
) -> list[Block]:
    """Pool text blocks together that are separated by
    code of size less or equal linecount"""

    def _is_code_block(b: TextBlock) -> bool:
        content: str = b.get_content()
        return (
            re.match(
                r"^```(\{[^\n]*\}|[^\n]*)?\n(.*?)\n```$",
                content,
                re.DOTALL,
            )
            is not None
        ) and (content.count('\n') <= (linecount + 1))

    return pool_textblocks_if(blocks, _is_code_block)


def pool_equation_blocks(blocks: list[Block]) -> list[Block]:
    """
    Pools text blocks together that are separated by equations
    """

    def _is_eq_block(block: TextBlock) -> bool:
        return (
            re.match(r"^\s*\$\$.*\$\$\s*$", block.get_content())
            is not None
        )

    return pool_textblocks_if(blocks, _is_eq_block)


def _find_largest_divisor(number: int, threshold: int):  # type:ignore
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


def pool_short_textblocks(
    blocks: list[Block], wordthresh: int = 120
) -> list[Block]:
    """
    Merges short text blocks together
    """

    if not blocks:
        return []

    # TO DO: use this to revise this function
    def _count_words(n: HeadingNode) -> int:  # type: ignore
        count: int = 0
        for c in n.children:
            if isinstance(c, TextNode):
                count += len(c.get_content().split())
        return count

    blocklist: list[Block] = []
    curblock: Block = blocks[0]  # assume a header
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
                    curblock = bl
            case _ as bl:  # reduce
                blocklist.append(curblock)
                curblock = bl
    blocklist.append(curblock)  # reduce

    return blocklist
