# Preparing a document for RAG

The function `markdown_rag` allows one to annotate a document with additional information prior to ingesting the document into a vector database. Annotations may be made by a language model automatically, then optionally inspected and edited, or made manually -- all types of interactions with the document, mediated by a language model or not, are possible.

The CLI interface to annotate a document by a language model is the following:

```python
from lmm.scan.scan_rag import markdown_rag, ScanOpts

document = markdown_rag(
    "MyDocument.md", 
    ScanOpts(summaries=True, questions=True), 
    "MyDocumentRagged.md"
)
```

This code snippet runs a language model to create summaries of content and questions answered by text in the metadata blocks annotating the headings of the markdown document "MyDocument.md". The changes are saved to "MyDocumentRagged.md". Alternatively, passing `True` as third argument in this call saves the changes directly to the original document. Summaries and questions are computed with the language model in the 'minor' section of Config.toml.

After processing the markdown file, it can be opened, inspected, and edited. Calling `markdown_rag` on an edited document again will not overwrite summaries and questions. To have the language model re-create a summary or a question, delete the summary or questions section in the metadata and run `markdown_rag` again.

NOTE: 
    The `markdown_rag` function will recompute summaries and questions if the text under one heading is changed. If you do not want summaries or questions to be recomputed after having changed the text under a heading, add the property `~freeze = True` to the metadata of the heading. Note that now, all RAG annotations of all subheadings of that heading will be frozen.

`ScanOpts` allows annotating the document with other properties. For example, `titles` adds the running titles of each heading. Other annotations that can be specified with `ScanOpts` are implementation steps that are not ususally of interest for manual inspection and editing (see the API section for an exhastive listing).

The summaries are created while respecting the hierarchical structure of the document. Summaries at higher level are written using as input text and summaries at lower levels.