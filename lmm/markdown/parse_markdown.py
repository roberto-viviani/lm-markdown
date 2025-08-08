"""This module contains a parser for a simplified version of Pandoc
Markdown, converting it into a list of blocks.

The parser creates a parse list of blocks defined by the following
types: HeaderBlock, MetadataBlock, HeadingBlock, TextBlock, and
ErrorBlock. Everything that is not parsed as a HeaderBlock,
MetadataBlock or HeadingBlock becomes a TextBlock.

The supported markdown is the same as in the pandoc specification
with the following exceptions:
- when used in the resto of the library, the document should start
    with a metadata block, which is parsed as a header
- title blocks marked by '%' are not supported (will be parsed as
    text blocks)
- metadata blocks are marked with three dashes, and contain a yaml
    specification
- metadata blocks must be preceded by a blank line only when they
    follow text
- setext-style headings are not supported (will be parsed as text
    blocks)
- a blank line is required before a heading only when the heading
    follows text
- horizontal rules with three dashes will be parsed as a metadata
    block delimiter
- headers of tables written with three dashes may be parsed as
    metadata blocks, if the table contains only one column

"""

# private use as protected
# pyright: reportPrivateUsage=false

# Definition of the grammar (informal). The lexemes are here entire
#     lines, not words
# document -> block [blank]+ block
# block  -> metadata | heading | content
# metadata -> metadata_marker metadata_content (metadata_marker |
#     metadata_end)
# metadata_marker   -> ---, optionally followed by comment following #
# metadata_content -> [content | blank]+
# metadata_end     -> ..., optionally followed by comment following #
# heading  -> #{1,6} followed by one space and the heading content
# blank    ->        # one or more blank lines
# content  -> .*     # everything else


from pathlib import Path
from typing import Tuple, Any, Callable
from pydantic import BaseModel
from typing_extensions import Literal

import re

from . import parse_yaml as pya
from .parse_yaml import MetadataDict
import yaml


# We define a discriminated union for the block types, in functional
# style, but we also add centralized handling of common functions,
# OOP-style.
class MetadataBlock(BaseModel):
    """This object represents the data of a metadata block in a
    markdown document.

    Important functions:
    serialize()     reconstitute a text repreentation of the metadata
    get_content()   the metadata
    get_key(key, default) a metadata value indexed by key
    """

    content: MetadataDict = {}
    comment: str = ""
    private_: list[object] = []
    type: Literal['metadata'] = 'metadata'

    def serialize(self) -> str:
        """A parsable textual representation of the block."""
        strrep = "---"
        if self.comment:
            strrep = strrep + " # " + self.comment
        # reconstitute original yaml block (see parse_yaml.py)
        content: str = pya.serialize_yaml_parse(
            (self.content, self.private_)
        )
        strrep = strrep + '\n' + content
        return strrep + "---\n"

    def get_info(self) -> str:
        """Printable block properties and content."""
        info = "\n-------------\nMetadata block"
        info += f" # {self.comment}\n" if self.comment else "\n"
        info += (
            pya.dump_yaml(self.content) if self.content else "<empty>"
        )
        if self.private_:
            info += "\n\nAdditional data:\n" + pya.dump_yaml(
                self.private_
            )
        return info

    def get_content(self) -> MetadataDict:
        """Returns a dictionary with the metadata."""
        return self.content

    def get_key(self, key: str, default: str = ""):
        """Returns the value of a key in the metadata."""
        return (
            self.content[key]
            if key in self.content.keys()
            else default
        )

    def deep_copy(self) -> 'MetadataBlock':
        return self.model_copy(deep=True)

    @staticmethod
    def _from_tokens(
        stack: list[Tuple['Token', str]],
    ) -> 'MetadataBlock | ErrorBlock':
        if not stack:
            # this is a programming error
            raise ValueError(
                "Invalid call to _from_tokens with empty list."
            )

        # check for comments
        comment_match = stack[0][1].strip().split('#', 1)
        comment = (
            comment_match[1].strip() if len(comment_match) > 1 else ''
        )

        # we assume the first and last tokens to be metadata markers
        content = '\n'.join([y for (_, y) in stack[1:-1]])

        # first use yaml parser, catching any error
        try:
            yamldata: Any = yaml.safe_load(content)
        except yaml.YAMLError as e:
            offending_meta = '\n'.join([y for (_, y) in stack])
            return ErrorBlock(
                content="\nYAML parse error in metadata block.",
                errormsg=str(e),
                origin=offending_meta,
            )

        # this returns the part of the yaml block that we want to
        # use here in 'part', the rest of the block in 'whole'.
        # See parse_yaml.py for explanation.
        try:
            part, whole = pya.split_yaml_parse(yamldata)
        except ValueError as e:
            # It is not clear if this error can occur.
            offending_meta = '\n'.join([y for (_, y) in stack])
            return ErrorBlock(
                content="\nUnexpected error when checking metadata.",
                errormsg=str(e),
                origin=offending_meta,
            )

        if (not part) and (not whole):
            return ErrorBlock(
                content="Invalid or empty metadata block.",
                origin="",
            )
        # We should be able to cope with this now
        # if not part:
        #     invalid_meta = '\n'.join([y for (_, y) in stack])
        #     return ErrorBlock(
        #         content="Metadata contains a dictionary or a list"
        #         + "of dictionaries that not acceptable for use in"
        #         + " LM markdown",
        #         origin=invalid_meta,
        #     )

        try:
            block = MetadataBlock(
                content=part, private_=whole, comment=comment
            )
        except Exception:
            # For the errors caught by pydantic
            try:
                block = MetadataBlock(
                    content={},
                    private_=[part] + whole,
                    comment=comment,
                )
            except Exception:
                return ErrorBlock(
                    content="Could not parse metadata:"
                    + " YAML object type not supported.",
                    errormsg="",  # a convoluted pydantic message
                    origin='\n'.join([y for (_, y) in stack]),
                )
        return block

    @staticmethod
    def _from_dict(
        dct: dict[object, object],
    ) -> 'MetadataBlock|ErrorBlock':
        if not pya.is_metadata_dict(dct):
            return ErrorBlock(
                content="Invalid dictionary to form metadata."
            )
        # now dct is a metadata dict
        return MetadataBlock(content=dct)  # type: ignore


class HeaderBlock(MetadataBlock):
    """This object represents the header block of a markdown document.
    It is the first block of the block list obtained from loading a
    markdown file with load_markdown.
    The behaviour of functions in this package when a header block is
    is inserted by code in a position other than the first is
    undefined.

    Important functions:
    serialize()     reconstitute a text repreentation of the metadata
    get_content()   the metadata
    get_key(key, default) a metadata value indexed by key
    """

    type: Literal['header'] = 'header'  # type: ignore

    def get_info(self) -> str:
        """Printable block properties and content."""
        info = "\n-------------\nHeader block"
        info += f" # {self.comment}\n" if self.comment else "\n"
        info += (
            pya.dump_yaml(self.content) if self.content else "<empty>"
        )
        if self.private_:
            info += "\n\nAdditional data:" + pya.dump_yaml(
                self.private_
            )
        return info

    def deep_copy(self) -> 'HeaderBlock':
        return self.model_copy(deep=True)

    @staticmethod
    def _from_metadata_block(
        block: MetadataBlock,
    ) -> 'HeaderBlock|ErrorBlock':
        if 'title' not in block.content:
            block.content['title'] = "Title"
        try:
            hblock = HeaderBlock(
                content=block.content,
                comment=block.comment,
                private_=block.private_,
            )
        except Exception as e:
            return ErrorBlock(
                content="Could not parse metadata:"
                + " YAML object type not supported.",
                errormsg=str(e),
            )
        return hblock

    @staticmethod
    def _from_tokens(
        stack: list[Tuple['Token', str]],
    ) -> 'HeaderBlock | ErrorBlock':
        block = MetadataBlock._from_tokens(stack)
        if isinstance(block, MetadataBlock):
            return HeaderBlock._from_metadata_block(block)
        return block

    @staticmethod
    def from_default(source: str = "") -> 'HeaderBlock':
        """Instantiate a default header block."""
        if not source:
            source = "Title"
        return HeaderBlock(content={'title': source})


class HeadingBlock(BaseModel):
    """This object represents a heading of the markdown document.
    A heading is a single line starting with one to six '#'
    characters followed by a space, and the title text.

    Important functions:
    serialize()     reconstitutes a text repreentation of the heading
    get_content()   the title given by the heading text
    """

    level: int
    content: str
    attributes: str = ""
    type: Literal['heading'] = 'heading'

    def serialize(self) -> str:
        """A parsable textual representation of the block."""
        strrep = "#" * self.level + " " + self.content
        if self.attributes:
            strrep = strrep + " {" + self.attributes + "}"
        return strrep + "\n"

    def get_info(self) -> str:
        """Printable block properties and content."""
        info = "\n-------------\nHeading block\n"
        info += str(self.content) if self.content else "<empty>"
        return info

    def get_content(self) -> str:
        """Returns the heading text"""
        return self.content

    def deep_copy(self) -> 'HeadingBlock':
        return self.model_copy(deep=True)

    @staticmethod
    def _from_tokens(
        stack: list[Tuple['Token', str]],
    ) -> 'HeadingBlock | ErrorBlock':
        # we assume that this is a single token with a heading content
        if len(stack) > 1:
            # throw error: this should not happen
            raise RuntimeError(
                "Unexpected token stack: Heading block should only "
                + "contain one line"
            )

        content = stack[0][1]

        # empty heading
        m = re.match(r'^#{1,6}\s*$', content)
        if m:
            return ErrorBlock(
                content="Empty heading content", origin=content
            )

        # check attributes: text delimited by '{' '}' at end of line
        m = re.search(r'\s+\{(.*?)\}\s*$', content)
        if m:
            # Extract the content before the attributes
            content = content[: m.start()].strip()
            attr_text = m.group(1).strip()
        else:
            attr_text = ""

        # parse heading at last, 1 to 6 '#' (guaranteed by
        # tokenization) followed by space and text
        m = re.search(r'^(#+)\s+(.+)', content)
        if not m:
            if attr_text:
                return ErrorBlock(
                    content="The heading specifies attributes, but "
                    + "there is no heading text",
                    origin=stack[0][1],
                )
            else:
                return ErrorBlock(
                    content="Cannot parse heading content",
                    origin=stack[0][1],
                )
        try:
            block = HeadingBlock(
                level=len(m.group(1)),
                content=m.group(2).strip(),
                attributes=attr_text,
            )
        except Exception as e:
            return ErrorBlock(
                content="Could not parse heading",
                errormsg=str(e),
                origin=stack[0][1],
            )
        return block


class TextBlock(BaseModel):
    """This object represents a text block from the markdown document.
    The text block starts after a heading, a metadata block, or a
    blank line, and ends with a blank line or the end of the document.

    Important functions:
    serialize()     reconstitutes a text representation of the block
    get_content()   returns a string with the text content
    extend()        extends the text with that of another text block
    """

    content: str
    type: Literal['text'] = 'text'

    def serialize(self) -> str:
        """A parsable textual representation of the block."""
        return self.content + "\n"

    def get_info(self) -> str:
        """Printable block properties and content."""
        info = "\n-------------\nText block\n"
        if self.content:
            content = self.content.split()
            if len(content) > 12:
                content = content[:11] + ["..."]
            info += " ".join(content) if content else "<empty>"
        else:
            info += "<empty>"
        return info

    def get_word_count(self) -> int:
        """Get the word count in the text block"""
        return len(self.content.split())

    def get_content(self) -> str:
        """Returns the text of the text block"""
        return self.content

    def is_empty(self) -> bool:
        return not self.content

    def extend(self, text: 'str | TextBlock') -> None:
        """Extend the content of the block with new text of with
        the content of another text block. The new content is
        added at the end of the block."""
        value: str
        match text:
            case str():
                value = text
            case TextBlock() as block:
                value = block.get_content()
        self.content = self.content + "\n\n" + value

    def deep_copy(self) -> 'TextBlock':
        return self.model_copy(deep=True)

    @staticmethod
    def _from_tokens(stack: list[Tuple['Token', str]]) -> 'TextBlock':
        # we assume that the first token is a content
        # and the last one is a blank line
        content = '\n'.join([y for (_, y) in stack[0:-1]])
        # we clear printed output of error blocks to allow re-scanning
        if content.startswith("** ERROR:"):
            return TextBlock(content="")

        return TextBlock(content=content)

    @staticmethod
    def from_text(text: str) -> 'TextBlock':
        """Instatiate a new text block from text."""
        return TextBlock(content=text)


class ErrorBlock(BaseModel):
    """This object represents a portion of the markdown document
    that gave rise to parsing errors.

    Important functions:
    serialize()     a textual representation of the error
    get_content()   the string with the error description
    self.origin     the markdown text that gave rise to the error
    """

    content: str = ""
    errormsg: str = ""
    origin: str = ""
    type: Literal['error'] = 'error'

    def serialize(self) -> str:
        """A textual representation of the error. When parsed, it will
        reconstitute the markdown text taht gave rise to the error."""
        content = "** ERROR: " + self.content + "**\n"
        if self.errormsg:
            content += self.errormsg + "\n"
        if self.origin:
            content += "\n" + self.origin + "\n\n"
        return content

    def get_info(self) -> str:
        """Printable block properties and content."""
        info = "\n-------------\nError block\n"
        info += self.content
        info = info if info else "empty error block"
        return info

    def get_content(self) -> str:
        """Returns the error message"""
        return self.content

    def deep_copy(self) -> 'ErrorBlock':
        return self.model_copy(deep=True)


Block = (
    MetadataBlock
    | HeaderBlock
    | HeadingBlock
    | TextBlock
    | ErrorBlock
)
# = Field(discriminator='type')

# Tokens are defined at the granularity of single lines.
from enum import Enum  # fmt: skip
Token = Enum(
    'Token',
    [
        ('METADATA_MARKER', 1),
        ('METADATA_END', 2),
        ('HEADING', 3),
        ('BLANK', 4),
        ('TEXT_CONTENT', 5),
        ('UNDEFINED', 0),
    ],
)


def _tokenizer(lines: list[str]) -> list[Tuple[Token, str]]:
    """Tokenize markdown lines into a sequence of token types and
    their content.

    This function processes each line of Markdown text and classifies
    it according to predefined token patterns. It applies regex
    patterns in a specific order to identify metadata markers,
    metadata end markers, headings, blank lines, and text content.
    The matching process is guaranteed to succeed because the
    TEXT_CONTENT token matches any line that doesn't match previous
    patterns.

    Args:
        lines: A list of strings, each representing a line from the
        markdown file

    Returns:
        A list of tuples, each containing a Token enum value and the
        original line text
    """

    token_patterns = [
        (r'^---(\s+#.*)?$', Token.METADATA_MARKER),
        (r'^\.{3}(\s+#.*)?$', Token.METADATA_END),
        (r'^(#{1,6})\s+(.*)', Token.HEADING),
        (r'^\s*$', Token.BLANK),
        (r'.*', Token.TEXT_CONTENT),
    ]
    regex_patterns = [(re.compile(x), y) for (x, y) in token_patterns]

    tokens: list[Tuple[Token, str]] = []
    for line in lines:
        for regex, token_type in regex_patterns:
            if regex.match(line):
                tokens.append((Token(token_type), line))
                break
    return tokens


def _parser(tokens: list[Tuple[Token, str]]) -> list[Block]:
    """Parse a list of tokens into a list of blocks.

    This function implements a simple state machine based on the first
    token of a potential block. The shift-reduce parser processes
    tokens (representing lines in a markdown file) by converting them
    into structured blocks.

    It takes over the following rules from pandoc markdown:
    - Headings must be preceded by a blank line when they follow a
        text block, otherwise they are treated as text
    - Text blocks are separated by blank lines
    - Metadata is defined by a '---' line at the start and '---' or
        '...' at the end
    - Multiple blank lines are treated as one.

    The parser handles different types of blocks:
    - HeaderBlock: The first block, if a metadata block
    - MetadataBlock: YAML blocks delimited by '---'
    - HeadingBlock: Section headings marked with '#'
    - TextBlock: Regular text content
    - ErrorBlock: Generated when parsing errors occur

    The parsing follows these rules:
    - Metadata blocks are reduced when a closing metadata marker is
        encountered
    - Heading blocks are reduced immediately
    - Text blocks are reduced when a blank line is encountered
    - If a metadata block, the first block is treated as the header
        block

    Args:
        tokens: A list of tuples, each containing a Token enum and the
            corresponding line text, as produced by _tokenizer

    Returns:
        A list of Block objects representing the parsed markdown
            content
    """

    document: list[Block] = []
    if len(tokens) == 0:
        return []

    # the shift-reduce loop.
    block_stack: list[Tuple[Token, str]] = []
    current_token: Token = Token.UNDEFINED
    for token_type, line in tokens:
        # shift the line onto the stack
        if token_type == Token.BLANK:
            # ignore blank lines at the beginning of blocks/multiple
            # blank lines
            if 0 == len(block_stack):
                continue
            if current_token == Token.BLANK:
                continue

        current_token = token_type
        block_stack.append((token_type, line))

        # check if we can reduce the block stack, matching the stack
        # based on the first token.
        match block_stack[0][0]:
            case Token.METADATA_MARKER:  # we are in a metadata block
                if (
                    current_token == Token.METADATA_END
                    or current_token == Token.METADATA_MARKER
                ):
                    if len(block_stack) == 1:
                        pass  # we just started
                    else:  # reduce
                        if 0 == len(document):
                            block = HeaderBlock._from_tokens(
                                block_stack
                            )
                        else:
                            block = MetadataBlock._from_tokens(
                                block_stack
                            )
                        document.append(block)
                        block_stack.clear()
                else:
                    pass  # keep the shift

            case Token.HEADING:  # can only occur as a single line
                document.append(
                    HeadingBlock._from_tokens(block_stack)
                )
                block_stack.clear()

            case Token.TEXT_CONTENT:  # we are in a text block
                if current_token == Token.BLANK:  # reduce
                    tbl: TextBlock = TextBlock._from_tokens(
                        block_stack
                    )
                    # empty blocks originate from cleared error output
                    if not tbl.is_empty():
                        document.append(tbl)
                    block_stack.clear()
                else:
                    pass  # keep the shift

            case _:
                pass  # keep the shift in all other cases

    if len(block_stack) > 0:
        match block_stack[0][0]:
            case Token.TEXT_CONTENT:  # final text block
                block_stack.append((Token.BLANK, ''))
                document.append(TextBlock._from_tokens(block_stack))
            case Token.METADATA_MARKER:  # dangling metadata
                block_stack.append((Token.BLANK, ''))
                document.append(
                    ErrorBlock(
                        content="Unclosed metadata block",
                        origin=(
                            "\n".join([b for (_, b) in block_stack])
                            if block_stack
                            else ""
                        ),
                    )
                )

            case _:
                # this should not happen
                raise RuntimeError(
                    "Unexpected token stack: " + str(block_stack)
                )

    return document


def parse_markdown_text(content: str) -> list[Block]:
    """Parse a pandoc markdown string into structured blocks.

    Args:
        content: a string containing markdown content.

    Returns:
        List of Block objects (HeaderBlock, MetadataBlock,
        HeadingBlock, TextBlock, ErrorBlock) representing
        the parsed content.

    Related functions:
        - serialize_blocks: Recreates Markdown text from blocks
        - blocklist_haserrors: Checks if parsing was successful
        - blocklist_errors: Returns list of error blocks
        - blocklist_get_info: Return string about the blocks
    """

    if not content:
        return []

    # preproc
    lines = (
        content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
    )

    # proc
    tokens = _tokenizer(lines)
    blocks = _parser(tokens)

    return blocks


def serialize_blocks(blocks: list[Block]) -> str:
    """Convert a list of Block objects to a markdown string.

    Joins the string representations of all blocks, adding blank lines
    between blocks as appropriate based on their types. No blank line
    is added after header blocks or before heading blocks.

    Args:
        blocks: List of Block objects to convert

    Returns:
        A string containing the markdown representation of the blocks
    """
    if not blocks:
        return ""

    last_block = blocks[0]
    content = last_block.serialize()
    trailing_block_types = (HeaderBlock, HeadingBlock, TextBlock)
    for block in blocks[1:]:
        if isinstance(last_block, trailing_block_types):
            content += '\n'
        content += block.serialize()
        last_block = block

    return (
        content[:-1]
        if content[-1] == "\n" and isinstance(last_block, TextBlock)
        else content
    )


def blocklist_copy(blocks: list[Block]) -> list[Block]:
    """Return a deep copy of the blocklist."""
    return [b.deep_copy() for b in blocks]


# info functions on block lists---------------------------------


def blocklist_errors(blocks: list[Block]) -> list[ErrorBlock]:
    """Return a list of errors in the block list."""
    return [
        block.deep_copy()
        for block in blocks
        if isinstance(block, ErrorBlock)
    ]


def blocklist_haserrors(blocks: list[Block]) -> bool:
    """Check if the block list contains errors."""
    for b in blocks:
        if isinstance(b, ErrorBlock):
            return True
    return False


def blocklist_map(
    blocks: list[Block],
    map_func: Callable[[Block], Block],
    filter_func: Callable[[Block], bool] = lambda _: True,
) -> list[Block]:
    """Apply map_func to all blocks that satisfy the predicate
    filter_func"""
    return [map_func(b.deep_copy()) for b in blocks if filter_func(b)]


def blocklist_get_info(blocks: list[Block]) -> str:
    """Collect info on all blocks in the list"""
    return "\n".join([x.get_info() for x in blocks])


# utilities-----------------------------------------------------


def load_blocks(source: str | Path) -> list[Block]:
    """Load a pandoc markdown file into structured blocks.

    Args:
        source: Path to a markdown file.

    Returns:
        List of Block objects (HeaderBlock, MetadataBlock,
            HeadingBlock, TextBlock, ErrorBlock) representing
            the parsed content.
    """

    # Load the markdown
    from .ioutils import load_markdown

    content = load_markdown(source)
    if not content:
        return []

    # Parse it
    blocks = parse_markdown_text(content)

    # Check for errors in the block list and log them to console
    from .ioutils import report_error_blocks

    report_error_blocks(blocks)

    # Returns all blocks, also error blocks
    return blocks


def save_blocks(file_name: str | Path, blocks: list[Block]) -> None:
    """Write a list of Block objects to a markdown file.

    Args:
        file_name: Path to the output file (string or Path object)
        blocks: List of Block objects to be serialized
    """
    from .ioutils import save_markdown

    content = serialize_blocks(blocks)
    save_markdown(file_name, content)


def save_blocks_debug(
    file_name: str | Path, blocks: list[Block], sep: str = ""
) -> None:
    """A debug version of save_blocks, with a separator string
    added to make clear where the block boundaries are"""

    from .ioutils import save_markdown

    content = ""
    for b in blocks:
        content += b.serialize()
        if isinstance(b, TextBlock):
            content += sep + "\n"
        content += "\n"

    save_markdown(file_name, content)
