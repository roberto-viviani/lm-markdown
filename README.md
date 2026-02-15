---
title: LM markdown
author: Roberto Viviani
date: 21.05.2025
---

This software extends markdown to enable interaction with a language model. The user exchanges information with the language model, which is turn is given information about the document that is being edited. This idea follows that concept of the notebook, where text and programmtic execution interact. The difference is that the programmatic execution takes place in the language model.

The exchange with the language model takes place through metadata blocks. Metadata blocks are a standard feature of much markdown, including Pandoc markdown and R markdown. However, the main use is providing a header to the document. In markdown, however, these blocks can be placed anywhere in the document. Internally, metadata blocks contain YAML specifications.

## Organization of documents and metadata blocks

A markdown document is organized with a header at the start (syntactically, a metadata block), and text ordered by headings (the level of the headings can range from 1 to 6). Text is organized in paragraphs (text blocks) separated by a blank line.

LM markdown add to this structure metadata blocks within the document. Metadata blocks annotate the document and refer to a portion of the document with the following rules:

- a metadata block always annotates the content that follows it
- a metadata block preceding a text block annotates the text block
- a metadata block preceding a heading annotates the text under the heading (including the text under the subheadings, if there are any)
- the header, itself a metadata block, annotates the whole document.

In the exchanges with the language model, which take place from within a metadata block, the user may refer to the text annotated by the metadata block as "the text". LM markdown provides this text to the language model, so that the model can use it to prepare its response. This model is flexible because the user may easily stake out a portion of text for the interaction with the language model by inserting temporary headings. When the interaction concerns a single block, the user may simply insert a metadata block before the text block.
