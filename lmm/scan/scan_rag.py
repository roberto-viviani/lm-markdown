"""
Operations on markdown blocks to prepare it for RAG (Retrieval
Augmented Generation) by enhancing it with metadata. This module
exemplifies how to use functions to change the markdown document
using its tree representation and higher-order traversal functions.

The operation that are supported by the module are

1. Validating the markdown structure and ensuring a proper header
    block
2. Adding unique IDs to blocks for tracking
3. Building hierarchical titles for headings based on document
    structure
4. Adding potential questions that sections of text answer using a
    language model
5. Adding summaries to heading nodes based on their content using
    a language model

This functionality is implemented by the utility functions:
    add_titles_to_headings
    add_id_to_nodes
    add_questions
    add_summaries

The functions scan_rag and markdown_rag use these functions to
carry out the operations as specified by an options record, ScanOpts.
The advantage of gathering these functions together in a superordinate
function is that this latter can be sure that the specifications are
consistent, and the functions are used in the right order.

Main superordinate functions:
    scan_rag: processes a blocklist adding metadata annotations
    markdown_rag: applies scan_rag to file
"""

import re
from pathlib import Path
from collections.abc import Callable
from pydantic import BaseModel, ConfigDict, Field, validate_call

# LM markdown
from lmm.markdown.ioutils import save_markdown
from lmm.utils.hash import generate_uuid, generate_random_string
from lmm.markdown.parse_markdown import (
    Block,
    HeaderBlock,
    HeadingBlock,
    TextBlock,
    ErrorBlock,
    blocklist_haserrors,
    blocklist_errors,
    blocklist_copy,
)
from lmm.markdown.blockutils import clear_metadata_properties
from lmm.markdown.tree import (
    MarkdownNode,
    HeadingNode,
    MarkdownTree,
    TextNode,
    blocks_to_tree,
    tree_to_blocks,
    pre_order_traversal,
    post_order_traversal,
)

# language models
from lmm.config.config import Settings, LanguageModelSettings
from lmm.language_models.langchain.runnables import (
    RunnableType,
    create_runnable,
)
from requests.exceptions import ConnectionError

# scan
from lmm.scan.scan import markdown_scan, blocklist_scan
from lmm.scan.scanutils import post_order_hashed_aggregation
from lmm.scan.scan_messages import blocklist_clear_messages
from lmm.scan.scan_keys import (
    DOCID_KEY,
    TEXTID_KEY,
    SUMMARY_KEY,
    HEADINGID_KEY,
    TITLES_KEY,
    SOURCE_KEY,
    QUESTIONS_KEY,
    UUID_KEY,
    OPTIONS_KEY,
    SKIP_KEY,
    WARNING_KEY,
    TITLES_TEMP_KEY,
)

from lmm.utils.logging import LoggerBase, get_logger

# Set up logger
logger: LoggerBase = get_logger(__name__)


class ScanOpts(BaseModel):
    """
    This options structure gathers the parameters for annotating
    the markdown (represented as a list of markdown blocks).
    All options default to no-op.

    Options:
        titles: add hierarchical titles to heading blocks
        questions: add potential questions to blocks
        questions_threshold: min word count to trigger questions
        summaries: add content summaries to heading blocks
        summaries_threshold: min word count to trigger summaries
        textid: adds a text id to text blocks
        headingid: adds a heading id to headings
        textUUID: adds a UUID to text blocks
        headingUUID: adds a UUID to headings

    Example of use:
        ```python
        opts = ScanOpts(titles = True) # add titles
        blocks = scan_rag(blocks, opts)
        ```
    """

    titles: bool = Field(
        default=False,
        description="Enable generation of hierarchical titles for "
        + "heading blocks based on document structure",
    )
    questions: bool = Field(
        default=False,
        description="Enable generation of potential questions that "
        + "text sections answer using language models",
    )
    questions_threshold: int = Field(
        default=15,
        gt=-1,
        description="Minimum word count threshold to trigger question"
        + " generation (ignored if questions=False)",
    )
    summaries: bool = Field(
        default=False,
        description="Enable generation of content summaries for "
        + "heading blocks using language models",
    )
    summary_threshold: int = Field(
        default=50,
        gt=-1,
        description="Minimum word count threshold to trigger summary "
        + "generation (ignored if summaries=False)",
    )
    remove_messages: bool = Field(
        default=False,
        description="Remove language model messages and metadata from"
        + " the processed document. Cleans up irrelevant metadata"
        + "created during interaction with the language model prior"
        + " to ingesting",
    )
    textid: bool = Field(
        default=False,
        description="Add unique text identifiers to text blocks for "
        + "tracking and reference in the vector database",
    )
    headingid: bool = Field(
        default=False,
        description="Add unique heading identifiers to heading blocks"
        + " for tracking and reference in the vector database",
    )
    textUUID: bool = Field(
        default=False,
        description="Add universally unique identifiers (UUIDs) to "
        + "text blocks for creation of id's in vector database",
    )
    headingUUID: bool = Field(
        default=False,
        description="Add universally unique identifiers (UUIDs) to "
        + "heading blocks for creation of group id's in vector database",
    )
    language_model_settings: (
        Settings | LanguageModelSettings | None
    ) = Field(
        default=None,
        description="A Settings object, a LanguageModelSettings "
        "object, or None. If provided, overrides settings in "
        "config.toml.",
    )

    model_config = ConfigDict(extra='forbid')


# function used to skip over when SKIP_KEY is true
def _filt_func(n: MarkdownNode) -> bool:
    """To use in aggregate function to exclude nodes with
    SKIP_KEY in metadata"""
    flag = bool(
        n.fetch_metadata_for_key(SKIP_KEY, False, False)
        if isinstance(n, HeadingNode)
        else n.fetch_metadata_for_key(SKIP_KEY, False)
    )
    return not flag


def blocklist_rag(
    blocks: list[Block],
    opts: ScanOpts = ScanOpts(),
    logger: LoggerBase = logger,
) -> list[Block]:
    """
    Prepares the blocklist structure for RAG (Retrieval Augmented
    Generation) by enhancing it with metadata.

    Args:
        blocks: a markdown block list
        opts: a ScanOpts object
        logger: a logger object (defaults to console logging)

    Returns:
        list[Block]: List of enhanced markdown blocks, or empty list
            if processing fails

    Note:
        The function adds several metadata fields to blocks:

        - docid: Unique document identifier
        - titles: Hierarchical heading path
        - textid/headingid: Unique block identifiers
        - questions: Potential questions answered by the text
        - summary: Content summaries for heading blocks

        The function will return an empty list if the input block
        list contains error blocks. It will add a default header
        if the header is missing, and a docid field to the header
        if this is missing.

    Raises ConnectionError, TypeError, ValueError, ValidationError

    Example of use:
        ```python
        opts = ScanOpts(titles = True) # add titles
        blocks = blocklist_rag(blocks, opts)

        # override language model from config.toml
        opts = ScanOpts(
            questions = True,               # add questions
            language_model_settings = LanguageModelSettings(
                model = "OpenAI/gpt-4o"
            )
        )
        blocks = blocklist_rag(blocks, opts)
        ```
    """

    # Validation
    build_titles = bool(opts.titles)
    build_questions = bool(opts.questions)
    build_summaries = bool(opts.summaries)
    build_textids = bool(opts.textid)
    build_headingids = bool(opts.headingid)
    build_textUUID = bool(opts.textUUID)
    build_headingUUID = bool(opts.headingUUID)
    if build_textUUID and (not build_textids):
        logger.info("bloclist_rag: text id's built to form UUID")
        build_textids = True
    if build_headingUUID and (not build_headingids):
        logger.info("blocklist_rag: text id's built to form UUID")
        build_headingids = True

    if not (
        build_titles
        or build_questions
        or build_summaries
        or build_textids
        or build_headingids
        or build_textUUID
        or build_headingUUID
    ):
        logger.info("No RAG changes specified for document.")
        return blocks

    # Validate for lm markdown
    blocks = blocklist_scan(blocks)

    # Further document validation
    if not blocks:
        raise RuntimeError(
            "Unreachable code reached: scan function "
            + "should not return an empty list"
        )
    if len(blocks) == 1 and isinstance(blocks[0], ErrorBlock):
        logger.error("Load failed:\n" + str(blocks[0].get_content()))
        return []
    if blocklist_haserrors(blocks):
        # convert markdown errors into logger errors
        for b in blocklist_errors(blocks):
            logger.error(b.get_content())
        logger.error("Errors in markdown. Fix before continuing.")
        return []

    # Preproc text blocks prior to annotations
    blocks = blocklist_copy(blocks)

    # this removes metadata properties that are not relevant
    # to RAG, such as chats
    if opts.remove_messages:
        blocks = blocklist_clear_messages(blocks)

    # Process directives
    root: MarkdownTree = blocks_to_tree(blocks, logger)
    if not root:
        return []
    logger.info("Processing " + root.get_content())

    # add titles for internal use
    add_titles_to_headings(root, logger, key=TITLES_TEMP_KEY)

    # add docid. This should identify the document uniquely.
    # It may be provided by the human user too.
    if DOCID_KEY not in root.metadata:
        # generate a random string to form doc id
        root.metadata[DOCID_KEY] = generate_random_string()

    # Add titles to headings
    if build_titles:
        logger.info("Adding titles to heading metadata.")
        add_titles_to_headings(
            root, logger, key=TITLES_KEY, filt_func=_filt_func
        )

    # Add an id to all heading and text blocks
    add_id_to_nodes(
        root,
        build_textids,
        build_headingids,
        root.get_metadata_string_for_key(DOCID_KEY),
        logger,
        filt_func=_filt_func,
    )

    # Add UUID to text and heading nodes
    def add_headingUUID_func(node: MarkdownNode) -> None:
        if not _filt_func(node):
            return
        if node.has_metadata_key(UUID_KEY):
            return
        if isinstance(node, HeadingNode):
            uuid_base: str | None = node.get_metadata_string_for_key(
                HEADINGID_KEY
            )
            if uuid_base is not None:
                node.set_metadata_for_key(
                    UUID_KEY,
                    generate_uuid(uuid_base),
                )
            else:
                # should not happen given we have generated TXTID's
                logger.warning("Could not set uuid for object")

    if build_headingUUID:
        logger.info("Adding UUIDs to hedings.")
        pre_order_traversal(root, add_headingUUID_func)

    def add_textUUID_func(node: MarkdownNode) -> None:
        if not _filt_func(node):
            return
        if node.has_metadata_key(UUID_KEY):
            return
        if isinstance(node, TextNode):
            uuid_base: str | None = node.get_metadata_string_for_key(
                TEXTID_KEY
            )
            if uuid_base is not None:
                node.set_metadata_for_key(
                    UUID_KEY,
                    generate_uuid(uuid_base),
                )
            else:
                # should not happen given we have generated TXTID's
                logger.warning("Could not set uuid for object")

    if build_textUUID:
        logger.info("Adding UUIDs to text blocks.")
        pre_order_traversal(root, add_textUUID_func)

    # Add source
    def add_source_func(node: MarkdownNode) -> None:
        if not _filt_func(node):
            return
        if isinstance(node, HeadingNode):
            if node.is_header_node():
                return
            source: str | None = node.fetch_metadata_string_for_key(
                DOCID_KEY
            )
            if source:
                node.metadata[SOURCE_KEY] = source

    pre_order_traversal(root, add_source_func)

    # Add a summary to heading nodes that is recomputed after changes
    if build_summaries:
        logger.info("Adding summaries about text.")
        add_summaries(root, opts, filt_func=_filt_func, logger=logger)

    # Add questions that the text answers, recomputed if text changes
    # (will use summaries if existing)
    if build_questions:
        logger.info("Adding questions about text.")
        add_questions(root, opts, filt_func=_filt_func, logger=logger)

    # check meta-data without text
    def _warn_empty_text(node: MarkdownNode) -> None:
        if node.is_header_node():
            pass
        elif isinstance(node, HeadingNode):
            if node.metadata:
                if len(node.get_text_children()) == 0:
                    node.metadata[WARNING_KEY] = (
                        "**Add text under this "
                        + "heading to avoid removal of "
                        + "metadata when ingesting**"
                    )
                elif WARNING_KEY in node.metadata:
                    node.metadata.pop(WARNING_KEY, "")
                else:
                    pass
        elif isinstance(node, TextNode):
            if node.metadata:
                if not node.get_content():
                    node.metadata[WARNING_KEY] = (
                        "**Add text under this "
                        + "metadata to avoid removal of "
                        + "metadata when ingesting**"
                    )
                elif WARNING_KEY in node.metadata:
                    node.metadata.pop(WARNING_KEY, "")
                else:
                    pass
        else:
            pass

    post_order_traversal(root, _warn_empty_text)

    # Re-create blocklist
    blocks = tree_to_blocks(root)

    # remove internal titles
    blocks = clear_metadata_properties(blocks, [TITLES_TEMP_KEY])

    return blocks


@validate_call(config={'arbitrary_types_allowed': True})
def markdown_rag(
    sourcefile: str | Path,
    opts: ScanOpts = ScanOpts(),
    save: bool | str | Path = True,
    *,
    max_size_mb: float = 50.0,
    warn_size_mb: float = 10.0,
    logger: LoggerBase = logger,
) -> list[Block]:
    """
    Scans the markdown file and adds information required for the
    ingestion in the vector database.

    opts defines what operations are conducted on the document,
    but if the header of the document contains an opts field,
    the specifications in the header are used.

    Args:
        sourcefile: the file to load the markdown from
        opts: a ScanOpts objects with the following options:
            titles (False)    add hierarchical titles to headings
            questions (False) add questions to headings
            questions_threshold (15) ignored if questions == False
            summaries (False) add summaries to headings
            summary_threshold (50) ignored if summaries == False
            remove_messages (False)
            textid (False)    add textid to text blocks
            headingid (False) add headingid to headings
            textUUID (False)  add UUID to text blocks
            headingUUID (False) add UUID to heading blocks
            pool_threshold (0) pooling of text blocks
        save: if False, does not save; if True, saves back to
            original markdown file; if a filename, saves to
            file. Defaults to True.
        max_size_mb: the max size, in MB, of the file to load
        warn_size_mb: the size of the input file that results in
            a warning
        logger: a logger object. Defaults to console logger.

    Returns:
        a list of blocks, starting with a header block.

    Note: if an error occurs and the blocklist becomes empty,
        it does not alter the source file.
    """

    blocks: list[Block] = markdown_scan(
        sourcefile,
        False,
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

    # Take over options if specified in header. The isinstance check
    # will always be true since markdown_scan provides a default
    # header if it is missing, but we check for pyright's benefit
    if isinstance(blocks[0], HeaderBlock):
        header: HeaderBlock = blocks[0]
        options: dict[str, object] = header.get_key_type(
            OPTIONS_KEY, dict, {}
        )
        if bool(options):
            logger.info("Reading opts specifications from header")
            try:
                # types checked and coerced by the pydantic model
                opts = ScanOpts(**options)  # type: ignore
            except Exception as e:
                logger.error(f"Invalid scan specification:\n{e}")
                return []
    else:
        raise RuntimeError(
            "Unreachable code reached: header block missing"
        )

    blocks = blocklist_rag(blocks, opts, logger)
    if not blocks:
        return []

    match save:
        case False:
            pass
        case True:
            save_markdown(sourcefile, blocks, logger)
        case str() | Path():
            save_markdown(save, blocks, logger)

    return blocks


def scan_rag(
    sourcefile: str | Path,
    opts: ScanOpts = ScanOpts(),
    save: bool | str | Path = True,
    *,
    titles: bool = False,
    questions: bool = False,
    questions_threshold: int = 15,
    summaries: bool = False,
    summary_threshold: int = 50,
    remove_messages: bool = False,
    max_size_mb: float = 50.0,
    warn_size_mb: float = 10.0,
    logger: LoggerBase = logger,
) -> None:
    """Carries out the interaction with the language model,
            returning a list of blocks with a header block first.

            opts defines what operations are conducted on the document,
            but if the header of the document contains an opts field,
            the specifications in the header are used.

            Args:
                sourcefile: the file to load the markdown from
    -           titles (False)    add hierarchical titles to headings
                questions (False) add questions to headings
                questions_threshold (15) ignored if questions == False
                summaries (False) add summaries to headings
                summary_threshold (50) ignored if summaries == False
                remove_messages (False)
                save: if False, does not save; if True, saves back to
                    original markdown file; if a filename, saves to
                    file. Defaults to True.
                max_size_mb: the max size, in MB, of the file to load
                warn_size_mb: the size of the input file that results in
                    a warning
                logger: a logger object. Defaults to console logger.

            Note:
                this is a stub of markdown_rag for interface bulding
    """

    try:
        opts = ScanOpts(
            titles=titles,
            questions=questions,
            questions_threshold=questions_threshold,
            summaries=summaries,
            summary_threshold=summary_threshold,
            remove_messages=remove_messages,
        )
        markdown_rag(
            sourcefile,
            opts,
            save,
            max_size_mb=max_size_mb,
            warn_size_mb=warn_size_mb,
            logger=logger,
        )
    except Exception as e:
        logger.error(str(e))


def add_titles_to_headings(
    root: MarkdownNode,
    logger: LoggerBase,
    *,
    key: str = TITLES_KEY,
    filt_func: Callable[[MarkdownNode], bool] = lambda _: True,
) -> None:
    """Recursively add titles to heading blocks in a markdown tree.

    This function maps nodes a markdown tree in a pre-order manner,
    collecting and concatenating the content of ancestor headings
    for each heading block. It adds a metadata field to HeadingBlock
    nodes, which represents the full hierarchical path of headings
    leading to that specific heading.

    Args:
        root: The root node of the markdown tree to process, or
            any other parent node
        filt_fun: A predicate the nodes must satisfy for titles to
            be added

    Note:
        - Only non-empty heading contents are included in the titles
        - The titles are added to the key field in the node's metadata
    """

    def map_func(node: MarkdownNode) -> None:
        # recursively add content of headings to key in metadata
        if isinstance(node, HeadingNode):
            if not filt_func(node):
                return
            if node.parent:
                titles: str = str(
                    node.parent.get_metadata_for_key(key, "")
                )
            else:
                titles = ""
            title: str = node.get_content()
            node.set_metadata_for_key(
                key,
                titles
                + (" - " if titles else "")
                + (title if title else ""),
            )
        return

    pre_order_traversal(root, map_func)


def add_id_to_nodes(
    root_node: MarkdownNode,
    textid: bool,
    headingid: bool,
    base_hash: str | None,
    logger: LoggerBase,
    *,
    filt_func: Callable[[MarkdownNode], bool] = lambda _: True,
) -> None:
    """Add unique identifiers to text and heading blocks in a markdown
    tree. These identifiers may be used when ingesting the document,
    to create the id's used by the vector database, such that new
    versions of the same blocks are overwritten in the database.

    This function traverses the markdown tree and assigns unique
    identifiers to TextBlock and HeadingBlock nodes. The identifiers
    are constructed using:
    1. A base hash derived from the document's title or a provided
        base_hash
    2. A sequential counter for text and heading blocks

    The function adds two types of metadata identifiers:
    - 'textid': Unique identifier for TextBlock nodes
    - 'headingid': Unique identifier for HeadingBlock nodes

    Args:
        root_node (MarkdownNode): The root node of the markdown tree
            to process
        base_hash (str, optional): A base hash to use for identifier
            generation. If not provided, a hash is generated from the
            root node's content (and will differ from that content).

    Identifier Format:
    - For text blocks: "{base_hash}.{sequential_number}"
      Example: "abc123.1", "abc123.2"
    - For heading blocks: "{base_hash}.h{sequential_number}"
      Example: "abc123.h1", "abc123.h2"

    Note:
        - Identifiers are _always_ added irrespective of whether they
            already exist in the node's metadata.
    """

    textid = bool(textid)
    headingid = bool(headingid)
    if not (textid or headingid):
        return

    if not base_hash:
        from lmm.utils.hash import base_hash as hash_func

        title: str = root_node.get_content()
        docid: str = root_node.get_metadata_string_for_key(  # type: ignore
            DOCID_KEY, title  # title is default if DOCID missing
        )
        # don't just use docid, make it same length and intuitively
        # not something to tamper with.
        base_hash = hash_func(docid)

    counter: dict[str, int] = {'text': 0, 'heading': 0}
    textkey = TEXTID_KEY
    headingkey = HEADINGID_KEY

    def _add_id(node: MarkdownNode) -> None:
        if not filt_func(node):
            return
        match node.block:
            case TextBlock() if textid:
                counter['text'] += 1
                node.metadata[textkey] = (
                    f"{base_hash}.{counter['text']}"
                )
            case HeadingBlock() if headingid:
                counter['heading'] += 1
                node.metadata[headingkey] = (
                    f"{base_hash}.h{counter['heading']}"
                )
            case _:
                pass

    pre_order_traversal(root_node, _add_id)


def add_questions(
    root: MarkdownNode,
    opts: ScanOpts,
    logger: LoggerBase,
    *,
    filt_func: Callable[[MarkdownNode], bool] = lambda _: True,
) -> None:
    """Add questions answered by text using a language model. Will
    not add qquestions to a header node.

    Args:
        root: a markdown node to start the traversal
        opts: options defining thresholds for computing summaries
        logger: a logger object
    """

    def llm_questions(text: str) -> str:
        if len(text.split()) < opts.questions_threshold:
            return ""
        response: str = ""
        try:
            kernel: RunnableType = create_runnable(
                "question_generator", opts.language_model_settings
            )
            response = kernel.invoke({'text': text})
        except ConnectionError:
            logger.error(
                "Could not connect to language models.\n"
                + "Check the internet connection."
            )
        except Exception:
            logger.error(
                "Error in using the language model to create "
                "questions."
            )

        # replace numbers
        pattern = r"\s*\d+[.)]?\s*"
        return re.sub(pattern, " - ", response)

    # use for development/debug purposes
    quest_func: Callable[[str], str] = lambda x: (  # type: ignore # noqa: 841
        "Questions this text answers" if x else ""
    )

    # do not call questions at header node.
    if root.is_header_node():
        for node in root.children:
            post_order_hashed_aggregation(
                node,
                llm_questions,
                QUESTIONS_KEY,
                filter_func=filt_func,
            )
    else:
        post_order_hashed_aggregation(
            root, llm_questions, QUESTIONS_KEY, filter_func=filt_func
        )


def add_summaries(
    root: MarkdownNode,
    opts: ScanOpts,
    logger: LoggerBase,
    *,
    filt_func: Callable[[MarkdownNode], bool] = lambda _: True,
) -> None:
    """Add summaries of text using a language model.

    Args:
        root: a markdown node to start the traversal
        opts: options defining thresholds for computing questions
        logger: a logger object
    """

    def llm_add_summary(text: str) -> str:
        if len(text.split()) < opts.summary_threshold:
            return ""
        response: str = ""
        try:
            kernel: RunnableType = create_runnable(
                kernel_name="summarizer",
                user_settings=opts.language_model_settings,
            )
            response = kernel.invoke({'text': text})
        except ConnectionError:
            logger.error(
                "Could not connect to language models.\n"
                + "Check the internet connection."
            )
        except Exception:
            logger.error(
                "Error in using the language model to create "
                "summaries."
            )

        return response

    # for development/debug
    summary_func: Callable[[str], str] = (  # type: ignore # noqa: 841
        lambda x: (
            f"Accumulated {len(x.split())} words."
            if len(x.split()) >= opts.summary_threshold
            else ""
        )
    )

    post_order_hashed_aggregation(
        root, llm_add_summary, SUMMARY_KEY, filter_func=filt_func
    )


def get_changed_titles(
    blocks: list[Block], logger: LoggerBase
) -> list[str]:
    """List the titles of all changed text. This is the
    headings that would be updated in a scan operation."""
    from ..markdown.treeutils import (
        get_nodes_with_metadata,
        get_headingnodes,
    )
    from .scan_keys import TXTHASH_KEY
    from ..utils.logging import LoglistLogger

    internal_logger = LoglistLogger()

    blocklist: list[Block] = blocklist_copy(blocks)
    root: HeadingNode | None = blocks_to_tree(blocklist, logger)
    if root is None:
        return []

    TITLES_KEY = "~__TITLES__"
    OUTPUT_KEY = "~__OUTPUT__"

    # check there are any hashes
    nodes: list[HeadingNode] = get_nodes_with_metadata(
        root, TXTHASH_KEY, HeadingNode
    )
    if not nodes:
        logger.info(
            "No hashes in document (the document has "
            "not been scanned yet)."
        )
        return []

    # add titles to report
    add_titles_to_headings(root, internal_logger, key=TITLES_KEY)
    if internal_logger.count_logs(level=2):
        for log in internal_logger.get_logs(level=2):
            logger.error(log)
        return []

    # execute hashing. Create output first as otherwise will
    # be reformed irrespective of hash.
    def _add_metadata_func(n: MarkdownNode) -> None:
        n.metadata[OUTPUT_KEY] = "fixed"

    post_order_traversal(root, _add_metadata_func)
    post_order_hashed_aggregation(
        root,
        lambda _: "changed",
        OUTPUT_KEY,
        True,
        filter_func=_filt_func,
        logger=internal_logger,
    )
    if internal_logger.count_logs(level=2):
        for log in internal_logger.get_logs(level=2):
            logger.error(log)
        return []

    # get nodes when hash discrepancy led to recompute
    nodes: list[HeadingNode] = get_headingnodes(
        root,
        True,
        lambda n: n.get_metadata_string_for_key(OUTPUT_KEY, "")
        == "changed"
        and not n.is_header_node(),
    )

    titles: list[str] = []
    for n in nodes:
        title: str | None = n.get_metadata_string_for_key(TITLES_KEY)
        if title:
            titles.append(title)

    return titles


if __name__ == "__main__":
    """Interactive loop to test module"""
    import sys
    from lmm.utils.ioutils import create_interface

    # this will not be used if the document has an options field
    # in the header
    opts = ScanOpts(questions=True, summaries=True)

    def main_f(x, y):  # type: ignore
        markdown_rag(x, opts, y)  # type: ignore
        return

    create_interface(main_f, sys.argv)  # type: ignore
