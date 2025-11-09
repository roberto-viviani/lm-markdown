"""
Operations on markdown blocks to interface with language models.

Main functions:
    scan_messages: looks for queries, messages, and edit prompts,
        and passes them to the language model, allowing the
        interaction
    markdown_messages: applies scan_messages to file
    remove_messages: removes message content from metadata
"""

from _collections_abc import dict_keys
from pathlib import Path
from pydantic import validate_call
from requests.exceptions import ConnectionError

# LM markdown
from lmm.markdown.parse_markdown import (
    Block,
    MetadataBlock,
    blocklist_haserrors,
    blocklist_copy,
)
from lmm.markdown.blockutils import clear_metadata_properties
from lmm.markdown.parse_yaml import MetadataValue
from lmm.markdown.tree import (
    MarkdownTree,
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
from lmm.utils.logging import LoggerBase, get_logger

from lmm.config.config import LanguageModelSettings
from lmm.language_models.tools import KernelNames
from lmm.language_models.langchain.runnables import (
    create_runnable,
    RunnableType,
)

from .scanutils import preproc_for_markdown

# metadata block keys
from .scan_keys import (
    CHAT_KEY,
    QUERY_KEY,
    MESSAGE_KEY,
    EDIT_KEY,
    SUMMARY_KEY,
)
from .scan import blocklist_scan, markdown_scan
from .scanutils import post_order_hashed_aggregation

# Set up logger
logger: LoggerBase = get_logger(__name__)


def _fetch_kernel(
    kernel_name: KernelNames, node: MarkdownNode | None = None
) -> RunnableType:
    """
    This function allows to use information from the metadata of a
    node to modify the properties of a language kernel loaded through
    the kernel module.
    At present, this is a stub to allow this future development, and
    the node argument is not used in the library.

    Raises:
        ValueError, TypeError, ValidationError
    """

    assert node is None

    if node is None:
        model: RunnableType = create_runnable(kernel_name=kernel_name)
    else:
        # unreacheable, future extension
        INCLUDE_HEADER = True
        model_properties: MetadataValue = node.fetch_metadata_for_key(
            str(kernel_name), INCLUDE_HEADER
        )
        if model_properties is None:
            model_properties = node.fetch_metadata_for_key(
                'model', INCLUDE_HEADER
            )
        if not isinstance(model_properties, dict):
            raise ValueError(
                "Invalid specification for language "
                + "model: must be a dictionary"
            )

        # type checks and coertions delegated to pydantic model
        settings = LanguageModelSettings(**model_properties)  # type: ignore
        model = create_runnable(
            kernel_name=kernel_name, user_settings=settings
        )
    return model


def _fetch_summary(node: MarkdownNode) -> str:
    """
    Utility to create a summary on a node with a heading model and
    return its content. The summary will be created after verifying
    that the text is changed, otherwise the old summary will be
    retained.

    Raises:
        ConnectionError, ValueError
    """
    model: RunnableType = _fetch_kernel(kernel_name="summarizer")
    post_order_hashed_aggregation(
        node,
        lambda x: model.invoke({'text': "\n".join(x)}),
        SUMMARY_KEY,
        True,  # i.e, hashed to TXTHASH_KEY
    )
    summary: str | None = node.get_metadata_string_for_key(
        SUMMARY_KEY, ""
    )
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


def _scan_queries(
    node: MarkdownNode, logger: LoggerBase = logger
) -> MarkdownNode:
    if QUERY_KEY not in node.metadata:
        return node

    # get the query
    query: str = str(node.metadata[QUERY_KEY])
    if not query:
        return node

    # check this query was not already answered
    if CHAT_KEY in node.metadata:
        chatvalues: MetadataValue = node.get_metadata_for_key(
            CHAT_KEY
        )
        chat: MetadataValue | list[MetadataValue] = (
            chatvalues
            if isinstance(chatvalues, list)
            else [chatvalues]
        )
        if chat and query in chat:
            idx: int = chat.index(query)
            if len(chat) - 1 > idx:
                return node

    # provide content depending on node type
    try:
        context: str = ""
        match node:
            case HeadingNode() as h:
                content: str = "\n".join(collect_text(h))
            case TextNode() as t:
                content = t.get_content()
                parent: HeadingNode | None = t.get_parent()
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
            model: RunnableType = _fetch_kernel("query_with_context")
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
        response = preproc_for_markdown(response)
    except ConnectionError as e:
        logger.error(
            "Could not connect to language models.\n"
            + "Check the internet connection."
        )
        return _add_chat(
            node, ["Error in connecting to the model", str(e)]
        )
    except Exception as e:
        logger.error(f"Error while messaging language model:\n{e}")
        return _add_chat(
            node, ["Error in language model exchange", str(e)]
        )

    return _add_chat(node, [query, response])


def _scan_messages(
    root: MarkdownNode, logger: LoggerBase = logger
) -> MarkdownNode:
    logger.warning("scan_messages not yet implemented.")
    return root


def _scan_edits(
    root: MarkdownNode, logger: LoggerBase = logger
) -> MarkdownNode:
    logger.warning("scan_edits not yet implemented.")
    return root


def _process_chain(
    root: MarkdownNode, logger: LoggerBase = logger
) -> MarkdownNode:
    root = pre_order_map_tree(
        root, lambda x: _scan_queries(x, logger)
    )
    root = pre_order_map_tree(
        root, lambda x: _scan_messages(x, logger)
    )
    root = pre_order_map_tree(root, lambda x: _scan_edits(x, logger))
    return root


def blocklist_messages(
    blocks: list[Block], logger: LoggerBase = logger
) -> list[Block]:
    """
    Carries out the interaction with the language model,
    returning a list of blocks with a header block first.

    Args:
        blocks: markdown blocks to process

    Returns:
        the processed list of blocks.
    """

    if not blocks:
        return []

    blocks = blocklist_scan(blocks)
    if blocklist_haserrors(blocks):
        logger.warning("Problems in markdown, fix before continuing")
        return blocks

    root: MarkdownTree = blocks_to_tree(
        blocklist_copy(blocks), logger
    )
    if not root:
        return []

    processed_root: MarkdownNode = _process_chain(root, logger)
    return tree_to_blocks(processed_root)


def blocklist_clear_messages(
    blocks: list[Block], keys: list[str] | None = None
) -> list[Block]:
    """Remove language model interactions from metadata. If specific
    keys are specified, only remove those keys.

    Args:
        blocks: the block list to handle
        keys (opts): specify the keys to remove. Otherwise, will
            remove the keys used in message exchanges.
    """

    if keys is not None:
        return clear_metadata_properties(blocks, keys)

    blocklist: list[Block] = []
    for b in blocks:
        if isinstance(b, MetadataBlock):
            newb: MetadataBlock = b.deep_copy()
            kks: dict_keys[str, MetadataValue] = newb.content.keys()
            if QUERY_KEY in kks:
                newb.content.pop(QUERY_KEY)
            if MESSAGE_KEY in kks:
                newb.content.pop(MESSAGE_KEY)
            if EDIT_KEY in kks:
                newb.content.pop(EDIT_KEY)
            if CHAT_KEY in kks:
                newb.content.pop(CHAT_KEY)

            if len(newb.content) > 0 or bool(newb.private_):
                blocklist.append(newb)
        else:
            blocklist.append(b)

    return blocklist


@validate_call(config={'arbitrary_types_allowed': True})
def markdown_messages(
    sourcefile: str | Path,
    save: bool | str | Path = True,
    *,
    max_size_mb: float = 50.0,
    warn_size_mb: float = 10.0,
    logger: LoggerBase = logger,
) -> list[Block]:
    """
    Carries out the interaction with the language model,
    returning a list of blocks with a header block first.

    Args:
        sourcefile: the file to load the markdown from
        save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file.

    Note:
        if an error occurs and the blocklist becomes empty,
        it does not alter the source file.
    """

    SAVE_FILE = False
    blocks: list[Block] = markdown_scan(
        sourcefile,
        SAVE_FILE,
        max_size_mb=max_size_mb,
        warn_size_mb=warn_size_mb,
        logger=logger,
    )
    if not blocks:
        return []
    if blocklist_haserrors(blocks):
        save_markdown(sourcefile, blocks, logger)
        logger.warning("Problems in markdown, fix before continuing")
        return []

    root: MarkdownTree = blocks_to_tree(
        blocklist_copy(blocks), logger
    )
    if not root:
        return []

    processed_root: MarkdownNode = _process_chain(root, logger)
    blocks = tree_to_blocks(processed_root)
    if not blocks:
        return []

    match save:
        case False:
            pass
        case True:
            save_markdown(sourcefile, blocks, logger)
        case str() | Path():
            save_markdown(save, blocks, logger)
        case _:  # ignore
            pass

    return blocks


@validate_call(config={'arbitrary_types_allowed': True})
def markdown_clear_messages(
    sourcefile: str | Path,
    keys: list[str] | None = None,
    save: bool | str | Path = True,
    logger: LoggerBase = logger,
) -> list[Block]:
    """
    Removes the messages from a markdown. If keys is specified,
    removes the metadata properties specified by keys.

    Args:
        sourcefile: the file to load the markdown from
        keys (optional): the keys of messages or any property to
            remove
        save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file.

    Note:
        if an error occurs and the blocklist becomes empty,
        it does not alter the source file.
    """

    SAVE_FILE = False
    blocks: list[Block] = markdown_scan(
        sourcefile, SAVE_FILE, logger=logger
    )
    if not blocks:
        return []

    if blocklist_haserrors(blocks):
        save_markdown(sourcefile, blocks, logger)
        logger.warning("Problems in markdown, fix before continuing")
        return []

    blocks = blocklist_clear_messages(blocks, keys)

    match save:
        case False:
            pass
        case True:
            save_markdown(sourcefile, blocks, logger)
        case str() | Path():
            save_markdown(save, blocks, logger)
        case _:  # ignore
            pass

    return blocks


def scan_messages(
    sourcefile: str | Path,
    save: bool | str | Path = True,
    *,
    max_size_mb: float = 50.0,
    warn_size_mb: float = 10.0,
    logger: LoggerBase = logger,
) -> None:
    """
    Carries out the interaction with the language model,
    returning a list of blocks with a header block first.

    Args:
        sourcefile: the file to load the markdown from
        save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file.

    Note:
        stub of markdown_messages for interface build
    """

    try:
        markdown_messages(
            sourcefile,
            save,
            max_size_mb=max_size_mb,
            warn_size_mb=warn_size_mb,
            logger=logger,
        )
    except Exception as e:
        logger.error(str(e))


def scan_clear_messages(
    sourcefile: str | Path,
    keys: list[str] | None = None,
    save: bool | str | Path = True,
    logger: LoggerBase = logger,
) -> None:
    """
    Removes the messages from a markdown. If keys is specified,
    removes the metadata properties specified by keys.

    Args:
        sourcefile: the file to load the markdown from
        keys (optional): the keys of messages or any property to
            remove
        save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file.

    Note:
        stub of markdown_clear_messages for interface build
    """

    try:
        markdown_clear_messages(sourcefile, keys, save, logger)
    except Exception as e:
        logger.error(str(e))


if __name__ == "__main__":
    """Interactive loop to test module"""
    import sys
    from lmm.utils.ioutils import create_interface

    create_interface(scan_messages, sys.argv)
