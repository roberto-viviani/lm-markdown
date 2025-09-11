---
title: LM markdown
author: Roberto Viviani
date: 21.05.2025
---

This software extends markdown to enable interaction with a language model. The user exchanges information with the language model, which is turn is given information about the document that is being edited. This idea follows that concept of the notebook, where text and programmtic execution interact. The difference is that the programmatic execution takes place in the language model.

The exchange with the language model takes place through metadata blocks. Metadata blocks are a standard feature of much markdown, including Pandoc markdown and R markdown. However, the main use is providing a header to the document. In markdown, however, these blocks can be placed anywhere in the document. Internally, metadata blocks contain YAML specifications.

There are three main ways in which the interaction takes place. In chat mode, the user and the language model exchange messages within the metadata blocks. To initiate an exchange with the language model, the user writes a line in a metadata block starting with '?: ', or a YAML property 'query: ' containing the text for the chat. The language model replies within the block. Further messages marked with '+: ', or YAML property 'message: ', continue the chat (to initiate a new chat, one deletes the chat from the block or marks the new exchande with 'query: ' or '?: '). In this and all other modalities, the metadata fields written by the language model are prefixed by a '~'. Fields written by the language model that are not for modification by the user are prefixed by '~~'.

In edit mode, the user requests the language model to edit part of the text. To initiate an edit exchange, the user puts a request in a metadata block starting with '=: ', or a YAML property 'edit: '. The model responds by creating a new heading for the old text (if there is any), ###### old text, and one for the new text ###### new text, with the new or the edited text.

In batch mode, a whole markdown document is scanned by the program and edited by the model. Sepcific code may be developed to provide edits (we refer to this as a 'batch model'). In the present case, a batch model was developed to prepare the text for ingestion in RAG. The code saves the edited markdown, and the user can inspect or edit, if necessary, the output. Batch mode is thought to allow repeated scans of the document, allowing rounds of interaction with the user. In the RAG batch model, properties are added in the metadata blocks (such as the questions the text answers). The user can edit, add, replace these properties. At successive scans, the properties are inserted by the batch model whenever they are missing (new text, for example). The convention, for batch model, is to mark inserted properties with the prefix '~' or '~~'.

In chat and edit mode, what the user writes in the metadata block are prompts concerning the text the block annotates. In batch mode, the prompts are part of the batch model.

## Organization of documents and metadata blocks

A markdown document is organized with a header at the start (syntactically, a metadata block), and text ordered by headings (the level of the headings can range from 1 to 6). Text is organized in paragraphs (text blocks) separated by a blank line.

LM markdown add to this structure metadata blocks within the document. Metadata blocks annotate the document and refer to a portion of the document with the following rules:

- a metadata block always annotates the content that follows it
- a metadata block preceding a text block annotates the text block
- a metadata block preceding a heading annotates the text under the heading (including the text under the subheadings, if there are any)
- the header, itself a metadata block, annotates the whole document.

In the exchanges with the language model, which take place from within a metadata block, the user may refer to the text annotated by the metadata block as "the text". LM markdown provides this text to the language model, so that the model can use it to prepare its response. This model is flexible because the user may easily stake out a portion of text for the interaction with the language model by inserting temporary headings. When the interaction concerns a single block, the user may simply insert a metadata block before the text block.

### Choice of the language model
The language model used by LM markdown is specified in environment variables. These variables must also contain the API keys of the language model one is using. 

LMM_SOURCE_EMBEDDING (default: "OpenAI")
LMM_MODEL_EMBEDDING  (default: "text-embedding-3-small")
LMM_SOURCE_MAJOR     (default: "OpenAI")
LMM_MODEL_MINOR      (default: "gpt-4o-mini")
LMM_SOURCE_MINOR     (default: same as LMM_SOURCE)
LMM_MODEL_MINOR      (default: depends on the source)
LMM_SOURCE_AUX       (default: same as LMM_SOURCE_MINOR)
LMM_MODEL_AUX        (default: same as LMM_MODEL_MINOR)

The specification will be overriden by a configuration file in TOML format located in the same folder as the entry python module.

### Functions working on the markdown

Functions that work on markdown files are the 'scan' functions, contained in a scan module or prefixed by 'scan_'. Current functions are

- scan: general overview of the markdown file, creates a header block if missing, notifies errors
- scan_messages: scans the metadata blocks for instructions of an interactive exchange with the language model
- scan_rag: scans the markdown and works on text to augment it prior to ingestion (see RAG below)

Some functions work by editing the metadata blocks. The field names that are supposed to be private to the software are prefixed by '~'. Fields that the user edits, and would like not to be touched by the software, can be postfixed by '='. To have a filed be recomputed, delete the field.

Generally, scan functions do not read settings but have arguments that may correspond to a setting value. (TODO: specify if reading settings options from metadata takes place at the level of scan functions.)

### The header
The header is a metadata block at the beginning of the document. It is a regular feature of R markdown, for example.

The header must contain a field 'title' (as in R markdown). If this field is absent, it is added by the software.

In LM markdown, the header may contain properties that override the choice of the language model specified in the environment variables:

- model:
  - embedding: default
  - major: openai/gpt-4.0-mini
  - minor: openai/gpt-4.1-nano
  - aux: something like mistral tiniest

The value 'default' means take the value from the configuration file or the environment. If these keys are missing, the values are automatically taken from the confuguration file or the environment. The sofwtare behaviour is undefined if no specification of the model is given in the header or in the environment.

The header may also contain a system message used in the interaction with the language model:

- model:
  -system: "You teach an undergraduate course in statistics using R."

If not specified, a default system message is used.

### RAG

The software allows interactive specification of additional fields that go into a RAG database. These fields are referred to below as 'anotations', and are:

questions: the questions that the text under the heading supports
summary: a summary of text under the heading.

The software automatically adds a 'title' field consisting of the concatenated titles of the heading and its parents.

The software provides a mechanism to automatically regenerate these fields if the text changes (by saving a hash)

The function scan_rag augments the markdown file with the additional fields. In edit mode, the markdown is saved byck to disk for the user to inspect it. The user can edit the fields proposed by the language model.

The RAG annotations are specified in the configuration file, and overridden in the header under model: RAG:, which may contain a list of the required fields to be computed.

## LM markdown API

The markdown API supports basic forms of interaction with the document, providing endpoints for the chat and the edit mode. For batch mode, the API supports basic forms of augmentation of the document that are typically required in preparing the document for RAG, such as providing summaries for each headings (with a overall summary in the header), providing the questions the text answers.

As the example of the summary functions may suggest, the API may exploit the hierarchical organization of text by considering text and the language model output recursively.

The API is thought to be extensible, starting from these basic ingredients, so as to support exchanges with the language model for specific use cases. For example, functions may be coded to support unstructured data analysis. These function may be interactively developed and applied to a document prior to deployment. Or, a library for data analysis may provide endpoints for the user to extract data from a document typology.

## Organization of code

The organization of code follows a typical data-centric program. 

A markdown file is read in and parsed to a list of blocks or three types: metadata (including the header), heading, and text. Serializing this list provides text that can be saved back to disk as a markdown file.

The block list is trasformed into a tree reflecting the hierachical organization of text implicit in the organization of text. The tree is composed of three types of nodes: the header node, which is the root of a tree representing a markdown document, heading nodes, which are parent nodes, and text nodes, which are leaf nodes containing text.

Each node may store metadata, which are taken from the metadata blocks that annotate that portion of the document. The header node is a heading node with the metadata of the header.

Traversing the tree in pre- or in post-order allows recursively processing text top-down or bottom-up. For example, bottom-up processing synthetises summaries of the docuemnt content by recursively creating summaries of summaries of the nodes lower in the hierachy. Most interactions with the language model turn out to be best executed when the document is represented as a tree.

The tree is retransformed into a list of blocks after changes or editing, and then from there serialized. The user does not see the tree; on-the-fly serialization gives the impression that the language model is editing the text the user sees in the editor directly.

To support rag, the block list is transformed into a list of document objects of the framework used to interface with the language model. This list is then processed to produce chunks and embeddings. This list is optionally re-transformed back into a markdown block list to be saved as a markdown file for inspection.

The processed list of documents is either ingested into a vector database supported by the framework, or transformed into a list of (jsonable) objects that the vector database understands (for example, qdrant has a point dataclass for this purpose).

### Parse markdown to a block list

The parser covers a simplified version of Pandoc markdown and converts a document into a list of "blocks": metadata, headings, and text. The parsing leaves the content of the text blocks unchanged. Unlike the pandoc parser, it maintains the position of the metadata blocks in the text. This parse list is a flat list to reflect the sequential nature of the markdown file.

The supported markdown is the same as in the pandoc specification with the following exceptions:
  - the document would start with a metadata block, which is the header of the document
  - title blocks marked by '%' are not supported (will be parsed as text blocks)
  - metadata blocks are marked with three dashes, and contain a yaml specification
  - metadata blocks must be preceded by a blank line only when they follow text
  - setext-style headings are not supported (will be parsed as text blocks)
  - a blank line is required before a heading only when the heading follows text
  - horizontal rules with three dashes will be parsed as a metadata block delimiter
  - headers of tables written with three dashes may be parsed as metadata blocks, if the table contains only one column
    
Errors arising during parsing are also reported as a special type of block (the error block) containing the error description.

#### parse_markdown

- parse_markdown_text. This function transforms a string (for example, the content a file previously read) into a list of markdown blocks. If the string starts with a metadata block, the block is made into the header block. In the code, the string is read using ioutils.load_markdown. To preserve the mapping from strings to block lists, the empty string gives an empty list.
- serialize. The opposite transformation.

#### ioutils

- load_markdown. Loads a utf-8 string file. This function provides a logger argument that may be used to process exceptions. The default prints the exception to the console.
- save_markdown. Saves a string to file, using the same logger interface.

### tree
This module provides the means to represent a block list as a tree. Note that the content of the blocks is not copied: in this sense, the tree is a representation of the document. Modifications on the tree may change the structure of the block list.

The tree contains two types of nodes: those that are parents to other nodes, and those that are just leaves, and contain text. The first node is the header node and is of the parent type; all other parent nodes are heading nodes, corresponding to a heading in the document. The heading level organizes the hierarchy between parent nodes.

While a block list can start with a text block, the tree must start with a header node, which is a metadata block with a title tag. This metadata block is parsed as a heading node with metadata given by the metadata block. All other metadata blocks are parsed as metadata of nodes given by parsing the block that follows: heading nodes form heading blocks, and text nodes from text blocks. A metadata block that is not followed by a non-metadata block is parsed as the metadata of an empty text node. Any non-metadata block that is not preceded by a metadata block is parsed as a node without metadata. 

Error blocks are parsed as text nodes with the text containing the invalid text. If unfolded into a block list, the original error block will now be a text block; but if this is serialized and parsed again, it will reproduce the error block.

This module contains the following:

- The definition of types representing the three types of nodes of the tree: HeaderNode, HeadingNode, and TextNode. These types contain member functions to facilitate access and modification of nodes.
- The transformation of a block list into a markdown tree. Metadata blocks become the metadata of the blocks that follow. Error blocks are transformed into text nodes. The inverse transformation unfolds the tree into a block list.
- Facilities to operate on the tree. These are higher-order functions that provide means to extract information from the tree or systematically modify the nodes. See the next sub-headings.

The module also contains utilities to load a tree from a markdown file, serialize the tree, and save it to a markdown file.

#### Markdown trees and subtrees
A heading node may be a singleton or have children, and may represent a subtree. A markdown tree may be initialized from a block list when the first block is a heading. If the first block is not a heading, a default heading must be created. A markdown tree is represented by a root heading node. A heading node cannot be null, but a markdown tree can:

MarkdownTree = HeadingNode | None

This preserves the isomorphism with a list, which can be empty (note however, that a list without a first heading block is not isomorphic to a tree). A subtree, if represented by a heading node cast to a tree, means that the subtree may be null.

For use in a rag model, the root heading node, representing the header, must have a 'title' and a 'docid' property.

# LM markdown for education

This adds special provisions for educational use of the markdown. This includes:

- providing summaries to headings that are indexed and retrievable by the student user
- providing means to enable the student user to retrieve a table of content/study plan
- providing means to enable the student user to produce study cards
- providing means to mark out exercises

Retrieval here is keyword-based, leveraging hybrid retrieval techniques.

It also uses a retrieval technique whereby text of headings up to a level are retrieved, leaving to the LMM the task of fishing out the content to be served to the student user. Here as well hybrid retrieval techniques make sure that relevant chapters are retrieved also based on hints that the rag author may insert for this purpose in appropriate metadata fields (~questions).

## Special text blocks for the RAG database

It will be possible to specify special types of text blocks to be ingested in the RAG database with a specific denomination. For example, it will be possible to add text marked as "exercise", or "study topics" that may be prepared for the exam. Also definitions, explanations not included in the main text but available in the non-linear interaction format allowed by RAG. These special text blocks will not be included in the "knitted" lecture notes, but will be available for the RAG interaction.

## Tranformation of the document for ingestion

LM markdown uses the tree structure of the document to annotate the document with additional information and, if required, prepare separate lists of "document" objects for search and for retrieval. This intermediate steps can be exported as a markdown to visualize the material that is being ingested in the index. Remember that text, not headings or metadata, are ingested.

The transformation proceeds in three steps. We will here refer to the list of markdown blocks as to the internal representation of the markdown, to "documents" as the format understood by the language model framework (for example, langchain), and to "points" as the format understood by the vector database (if it is used directly instead of being intermediated by the framework).

In a first step, a preliminary evaluation of the text is made. This includes pooling text blocks that contain equations only and very short text blocks, so that context is not lost. A policy could alos be implemented for code blocks.

A second optional step identifies the larger documents that will be retrieved. These could be -- depending on the structure of the headings -- the text under each headings, or the text splitted across text blocks if a given size threshold has been reached. Usually, this step will consist of establishing an adequate document size that will be returned to the language model at retrieval. A final part of this step specifies the annotations that will be computed at this level of granularity on the documents.

A third and final step computes the "chunks" of text by splitting the text from the second step and compute the embeddings (it only makes sense if the level of granularity of the text is smaller than in the previous step). It is always possible to use a no-op chunking, but this is not foreseeen as a normal strategy. To do this, the list of markdown blocks must be transformed into a list of "documents" as understood by the language model framework. Note, however, that before doing this transformation it will be necessary to collect all metadata that go into the embedding, or compute the relevant annotations.

Apart from the first step, the operations of the transformations are customizable. The user can specify the transformation options in the header. The intermediate output is a list of markdown blocks that may be saved as a markdown file for inspections. The final output is a list of "points" that the vector database can ingest (or of "documents" if a language model framework is used as intermediary).

Note that the transformation of a markdown list block will result in one list of "documents" only if the second step is missing. If not, the fransformation returns two lists: the list of the larger documents for retrieval, and the smaller documents for embedding and search.

In addition, some annotations may be transformed into text to be appended to the document lists. This is the case for summary annotations.

From an implementational point of view, this scheme corresponds to the "strategy" pattern. In a functional style, a record of functions or options (i.e., an associative array) may be enough.

## RAG with QDRANT

The document organization for ML markdown for education includes the following properties:

- a list of questions that the text answers
- titles
- summary of text under headings

These properties are retrieved before the text is chunked. The document and chunks, therefore, can be indexed using different strategies. 

### Questions and titles:

- none present: only text is available for embedding.
- either or both present:
  - merged with text and indexed as a whole block
  - merged together and indexed, sent as a "multivector" to the database together with the separate index of the text
  - added as a sparse vector (for user of same language), with text as a dense vector for hybrid search
  - added as a sparse vector (for user of same lanugage), with the merged variants listed above for hybrid search

For example, a strategy would be a hybrid query on a multivector to represent questions and titles in boch sparse and dense representations.

### Summaries

Indexed as autonomous blocks (see raptor paper)

### Pooling of text

The document organization allows to retrieve text prior to chunking. Specifically, it is possible to merge the text under a heading, or to merge the text under a heading under a word count limit (see step 2 of document transformation). This text may be stored in a separate collection and retrieved using the 'group_by' and 'lookup' query options. This is a departure from classic RAG strategies, but may be advantageous since now there are hardly limits in the context memory, and the model may be tasked to fish out the relevant information better than just embedding. That is, embedding is used only to make sure to retrieve all relevant texts.

- based on overlapped chunking (standard strategy)
- form pooled text that is retrieved when a chunk is selected

### Types of material

In addition to text, the following type of text may be explicitly marked by a descriptor in the payload.

- summaries (arising from previous step)
- exercises
- topics included in exams
- whatever else the instructor may want to include

This special material is marked by a value in a 'type' keyword in the payload, and indexed in the main collection.

The user is told that this material may be selectively retrieved by using a keyword in the query. For example, *summary*, or *exercise*. If this keyword is detected, then the software computes a prefetch with a large limit, and applies boosting on the type keyword indicated by the user. If the keyword is found, the prompt to the model can be modified to indicate this fact.