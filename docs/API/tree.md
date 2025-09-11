# Tree representation of markdown documents

The tree representation of a markdown document is a graph representing a hierarchy, in which all nodes, except the root node, have one and only one parent (out-tree). When representing a markdown document, the header is used to create the root node. All headings are children of the root or of other headings, depending on the level of the heading, like the titles of chapters and sections in a book. The text blocks of the markdown become the children of the headings where the text is contained.

In LM markdown, a default header is created if the document has none when a tree is formed. Metadata are attached to the heading or text block that follows. Metadata without following text are attached to a created text block without content. Because of these modifications, unfolding the tree hieararchy and recreating the file may not reproduce the original document.

## Tree creation
::: lmm.markdown.tree

# Tree utils
::: lmm.markdown.treeutils
