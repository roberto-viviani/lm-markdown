# Interactive work on markdown file

LM markdown supports interactive writing through the interaction with a language model. This interaction takes place through metadata blocks that may be inserted at any point in the document. You can ask a question about the text, or request work on the text, by posting a message to the language model in the metadata.

## Metadata and hierarchical structure

When you interact with the language model, this latter automatically receives parts of the text of the markdown file as context to the query. This text is selected based on the position of the metadata block where the query is formulated.

A general rule in LM markdown is that metadata blocks annotate the material that follows. If what follows a metadata block is a heading, then the annotation refers to the whole content under the heading. If what follows the block is a text block, then the annotation only refers to the following text block.

## Example of getting feedback from a language model

Here is an example of markdown text, with a heading and some text that follows.

```markdown
## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used
in practice. Here, we will look at linear models from a practical perspective, emphasizing the
issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the *predictors* (also known 
as *independent variables*), and an *outcome* variable (or *dependent variable*). In the
simplest setting, this association may be approximated and displayed by a line relating a
predictor and the outcome. However, the generalizations of linear models allow capturing much
more complex associations.
```

To receive feedback about this text, we will insert a metadata block before the heading, and interrogate the language model through a key/value pair, where the key is 'query' or simply '?':

```markdown
---
query: please review the text for clarity and conciseness.
---
## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used
in practice. Here, we will look at linear models from a practical perspective, emphasizing the
issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the *predictors* (also known 
as *independent variables*), and an *outcome* variable (or *dependent variable*). In the
simplest setting, this association may be approximated and displayed by a line relating a
predictor and the outcome. However, the generalizations of linear models allow capturing much
more complex associations.
```

Because this query is placed before the heading, both text blocks will be sent to the language model for it to respond to the query. In contrast, had the metadata block been placed before a text block, only the text of that block would be sent to the language model. Also note that 'text' is the term to use to refer to the text that the language model receives during the query.

The interaction with the language model is controlled thorugh the CLI with a call to `markdown_messages`:

```python
from lmm.scan.scan_messages import markdown_messages

document = markdown_messages("MyText.md")
```

The edited text is automatically reloaded in a markdown text editor such as MS Visual Studio Code or RStudio, with the response of the language model listed at the bottom of the metadata block:

```markdown
---
query: please review the text for clarity and conciseness.
~chat:
- please review the text for clarity and conciseness.
- Here is a revised version of the text for improved clarity and conciseness:

"Linear models and their generalizations make up most statistical models used in practice.
This discussion focuses on applying linear models correctly and interpreting their output. 
Linear models describe the relationship between predictors (independent variables) and an 
outcome (dependent variable). In the simplest case, this relationship can be represented by a
line between a predictor and the outcome. More advanced linear models can capture more complex
associations."

Let me know if you would like it further simplified or adjusted!
---

## What are linear models?

Linear models and their generalizations constitute the majority of the statistical models used
in practice. Here, we will look at linear models from a practical perspective, emphasizing the
issues in correctly applying them and in understanding their output.

Linear models capture the association between a set of variables, the *predictors* (also known 
as *independent variables*), and an *outcome* variable (or *dependent variable*). In the
simplest setting, this association may be approximated and displayed by a line relating a
predictor and the outcome. However, the generalizations of linear models allow capturing much
more complex associations.
```

The interaction with the language model can be continued by re-writing the text of the query. The application checks that the query is new, and sends it to the language model. If you delete che ~chat in the meatadata block without deleting the query: key, then the query (and the text) will be sent to the language model again.

It is possible to save the edited markdown to a different file. In this case, indicate the new file name as the second argument of the call to `markdown_messages`:

```python
from lmm.scan.scan_messages import markdown_messages

document = markdown_messages("MyText.md", "MyTextEdited.md")
```

## Remove messages to and from the language model

To eliminate all messages arising from interactions with language models from the markdown, use `markdown_remove_messages`.

Messages to and from the language model are automatically removed when the document is scanned for RAG annotations.
