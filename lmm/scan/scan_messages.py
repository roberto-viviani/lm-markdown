"""
Operations on markdown blocks to interface with language models.

Main functions:
    scan_messages: looks for queries, messages, and edit prompts,
        and passes them to the language model, allowing the
        interaction
    markdown_messages: applies scan_messages to file
    remove_messages: removes message content from metadata
"""

# pyright: basic

from pathlib import Path

# LM markdown
from lmm.markdown.parse_markdown import (
    Block,
    MetadataBlock,
    blocklist_haserrors,
    blocklist_copy,
)
from lmm.markdown.parse_yaml import MetadataValue
from lmm.markdown.tree import (
    MarkdownNode,
    HeadingNode,
    TextNode,
    blocks_to_tree,
    tree_to_blocks,
)
from lmm.markdown.treeutils import (
    collect_text,
    pre_order_map_tree,
)
from lmm.markdown.ioutils import save_markdown
import lmm.scan.scan as scan

# metadata block keys
from .scan_keys import (
    CHAT_KEY,
    QUERY_KEY,
    MESSAGE_KEY,
    EDIT_KEY,
    SUMMARY_KEY,
)

from lmm.config.config import LanguageModelSettings
from lmm.language_models.kernels import KernelNames
from lmm.language_models.langchain.kernel import (
    create_kernel,
    KernelType,
)

from requests.exceptions import ConnectionError

from lmm.utils.logging import get_logger

# Set up logger
logger = get_logger(__name__)


def _fetch_kernel(
    kernel_name: KernelNames, node: MarkdownNode | None = None
) -> KernelType:
    """This function allows to use information from the metadata of a
    node to modify the properties of a language kernel loaded through
    the kernel module.
    At present, this is a stub to allow this future development, and
    the node argument is not used in the library.

    Raises:
        ValueError, TypeError, ValidationError
    """

    assert node is None

    if node is None:
        model = create_kernel(kernel_name=kernel_name)
    else:
        INCLUDE_HEADER = True
        model_properties: MetadataValue = node.fetch_metadata_for_key(
            str(kernel_name), INCLUDE_HEADER
        )
        if not isinstance(model_properties, dict):
            raise ValueError(
                "Invalid specification for language "
                + "model: must be a dictionary"
            )

        # type checks and coertions delegated to pydantic model
        settings = LanguageModelSettings(**model_properties)  # type: ignore
        model = create_kernel(
            kernel_name="summarizer", user_settings=settings
        )
    return model


def _fetch_summary(node: MarkdownNode) -> str:
    """Utility to create a summary on a node with a heading model
    and return its content.
    Raises:
        ConnectionError
    """
    model = _fetch_kernel(kernel_name="summarizer")
    scan.post_order_aggregation(
        node,
        lambda x: model.invoke({'text': "\n".join(x)}),
        SUMMARY_KEY,
    )
    summary = node.get_metadata_string_for_key(SUMMARY_KEY, "")
    return summary if summary else ""


def _add_chat(node: MarkdownNode, chat: list[str]) -> MarkdownNode:
    value: MetadataValue = node.get_metadata_for_key(CHAT_KEY, None)
    # These types are only for compatibility. We only save strings in
    # CHAT_KEY.
    oldchat: list[str | int | float | bool] = []
    match value:
        case str() as v:
            oldchat.append(v)
        case list() as v:
            for val in value:
                oldchat.append(str(val))
        case None:  # there is no chat
            pass
        case _:
            raise RuntimeError(
                "Unreachable code reached: "
                + "invalid chat list data"
            )

    node.set_metadata_for_key(CHAT_KEY, oldchat + chat)
    return node


def _scan_queries(node: MarkdownNode) -> MarkdownNode:
    if QUERY_KEY not in node.metadata:
        return node

    # get the query
    query: str = str(node.metadata[QUERY_KEY])
    if not query:
        return node

    # check this query was not already answered
    if CHAT_KEY in node.metadata:
        chatvalues = node.get_metadata_for_key(CHAT_KEY)
        chat = (
            chatvalues
            if isinstance(chatvalues, list)
            else [chatvalues]
        )
        if chat and query in chat:
            idx = chat.index(query)
            if len(chat) - 1 > idx:
                return node

    # provide content depending on node type
    try:
        context: str = ""
        match node:
            case HeadingNode() as h:
                content = "\n".join(collect_text(h))
            case TextNode() as t:
                content = t.get_content()
                parent = t.get_parent()
                if parent:
                    context = _fetch_summary(parent)
            case _:
                # this should not happen
                raise ValueError(
                    "Unrecognized node type in summarization request"
                )
    except ConnectionError as e:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection."
        )
        return _add_chat(
            node, ["Error in connecting to the model", str(e)]
        )
    except Exception as e:
        logger.error(
            "Error in using the language model to create summaries."
        )
        return _add_chat(
            node,
            [
                "Error in using language model to create summaries",
                str(e),
            ],
        )

    # ask model
    try:
        response: str
        if context:
            model = _fetch_kernel("query_with_context")
            response = model.invoke(
                {
                    'query': query,
                    'text': content,
                    'context': context,
                }
            )
        else:
            model = _fetch_kernel("query")
            response = model.invoke(
                {
                    'query': query,
                    'text': content,
                }
            )
        # encode newline and avoid conflict with
        # metadata header markers
        if "'" in response:
            response = response.replace('"', '')
            response = '"' + response + '"'
        response = (
            response.replace('\n---', '\n----')
            .replace('\n...', '\n....')
            .replace('\n', '__NEWLINE__')
        )
    except ConnectionError as e:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection."
        )
        return _add_chat(
            node, ["Error in connecting to the model", str(e)]
        )
    except Exception as e:
        return _add_chat(
            node, ["Error in language model exchange", str(e)]
        )

    return _add_chat(node, [query, response])


def _scan_messages(root: MarkdownNode) -> MarkdownNode:
    return root


def _scan_edits(root: MarkdownNode) -> MarkdownNode:
    return root


def _process_chain(root: MarkdownNode) -> MarkdownNode:
    root = pre_order_map_tree(root, _scan_queries)
    root = pre_order_map_tree(root, _scan_messages)
    root = pre_order_map_tree(root, _scan_edits)
    return root


def scan_messages(blocks: list[Block]) -> list[Block]:
    """Carries out the interaction with the language model,
    returning a list of blocks with a header block first.

    Args:
        blocks: markdown blocks to process

    Returns: the processed list of blocks.
    """

    if not blocks:
        return []

    blocks = scan.scan(blocks)
    if blocklist_haserrors(blocks):
        logger.warning("Problems in markdown, fix before continuing")
        return blocks

    root = blocks_to_tree(blocklist_copy(blocks))
    if not root:
        return []

    root = _process_chain(root)
    return tree_to_blocks(root)


def remove_messages(blocks: list[Block]) -> list[Block]:
    """Remove language model interactions from metadata"""

    blocklist: list[Block] = []
    for b in blocks:
        if isinstance(b, MetadataBlock):
            newb: MetadataBlock = b.deep_copy()
            keys = newb.content.keys()
            if QUERY_KEY in keys:
                newb.content.pop(QUERY_KEY)
            if MESSAGE_KEY in keys:
                newb.content.pop(MESSAGE_KEY)
            if EDIT_KEY in keys:
                newb.content.pop(EDIT_KEY)
            if CHAT_KEY in keys:
                newb.content.pop(CHAT_KEY)
            blocklist.append(newb)
        else:
            blocklist.append(b)

    return blocklist


def markdown_messages(
    sourcefile: str | Path, save: bool | str | Path = False
) -> list[Block]:
    """Carries out the interaction with the language model,
    returning a list of blocks with a header block first.

    Args: source: the file to load the markdown from
          save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file.

    Returns: the processed list of blocks.

    Note: if an error occurs and the blocklist becomes empty,
        it does not alter the source file.
    """

    blocks = scan.markdown_scan(sourcefile, False)
    if not blocks:
        return []
    if blocklist_haserrors(blocks):
        # markdown_scan(sourcefile, True)
        save_markdown(sourcefile, blocks)
        logger.warning("Problems in markdown, fix before continuing")
        return blocks

    root = blocks_to_tree(blocklist_copy(blocks))
    if not root:
        return []

    root = _process_chain(root)
    blocks = tree_to_blocks(root)
    if not blocks:
        return []

    match save:
        case False:
            pass
        case True:
            save_markdown(sourcefile, blocks)
        case str() | Path():
            save_markdown(save, blocks)
        case _:  # ignore
            pass

    return blocks


if __name__ == "__main__":
    """Interactive loop to test module"""
    import sys
    from lmm.utils.ioutils import create_interface

    create_interface(markdown_messages, sys.argv)
