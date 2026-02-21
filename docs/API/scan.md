# Interaction with language model

This software extends markdown to enable interaction with a language model. The user exchanges information with the language model, which is turn is given information about the document that is being edited. This idea follows that concept of the notebook, where text and programmtic execution interact. The difference is that the programmatic execution takes place in the language model.

The exchange with the language model takes place through metadata blocks. Metadata blocks are a standard feature of much markdown, including Pandoc markdown and R markdown. However, the main use is providing a header to the document. In markdown, however, these blocks can be placed anywhere in the document. Internally, metadata blocks contain YAML specifications.

There are three main ways in which the interaction takes place. In chat mode, the user and the language model exchange messages within the metadata blocks. To initiate an exchange with the language model, the user writes a line in a metadata block starting with '?: ', or a YAML property 'query: ' containing the text for the chat. The language model replies within the block. Further messages marked with '+: ', or YAML property 'message: ', continue the chat (to initiate a new chat, one deletes the chat from the block or marks the new exchande with 'query: ' or '?: '). In this and all other modalities, the metadata fields written by the language model are prefixed by a '~'. Fields written by the language model that are not for modification by the user are prefixed by '~~'.

In edit mode, the user requests the language model to edit part of the text. To initiate an edit exchange, the user puts a request in a metadata block starting with '=: ', or a YAML property 'edit: '. The model responds by creating a new heading for the old text (if there is any), ###### old text, and one for the new text ###### new text, with the new or the edited text.

In batch mode, a whole markdown document is scanned by the program and edited by the model. Sepcific code may be developed to provide edits (we refer to this as a 'batch model'). The code saves the edited markdown, and the user can inspect or edit, if necessary, the output. Batch mode is thought to allow repeated scans of the document, allowing rounds of interaction with the user. In the RAG batch model, properties are added in the metadata blocks (such as the questions the text answers). The user can edit, add, replace these properties. At successive scans, the properties are inserted by the batch model whenever they are missing (new text, for example).

In chat and edit mode, what the user writes in the metadata block are prompts concerning the text the block annotates. In batch mode, the prompts are part of the batch model.

# Generic scan module
::: lmm.scan.scan

# Scan module for LLM interaction
::: lmm.scan.scan_messages

# Scan module for RAG
::: lmm.scan.scan_rag

# Scan module for block splitting
::: lmm.scan.scan_split

# Scan module for Chunks
::: lmm.scan.chunks

# Scan module for Scan Keys
::: lmm.scan.scan_keys

# Scan module for Scan Utilities
::: lmm.scan.scanutils
