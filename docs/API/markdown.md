# Load and save markdown documents

## Load and save module: `ioutils`
::: lmm.markdown.ioutils

# Parse markdown into block lists

Markdown documents are parsed into lists of `block` objects. These objects may be of three types: metadata (including the header, `MetadataBlock`), heading (`HeadingBlock`), and text (`TextBlock`). Serializing this list provides text that can be saved back to disk as a markdown file.

The parser covers a simplified version of Pandoc markdown. The parsing leaves the content of the text blocks unchanged. Unlike the pandoc parser, it maintains the position of the metadata blocks in the text. This parse list is a flat list to reflect the sequential nature of the markdown file.

::: lmm.markdown.parse_markdown

# Utilities to work with block lists: `blockutils`

::: lmm.markdown.blockutils

# Yaml parser

::: lmm.markdown.parse_yaml

