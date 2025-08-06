---  # This is a comment
title: My Document
author: John Doe
internal:
  first: 1
  second: 2
---

There is some text after the header without a preceding title.

--- # questions already in original ("The question of the day")
~questions: The question of the day
---
# Initial first-level heading
This is text located immediately below the first-level heading without a blank line in between.

## Second level heading {class = "elicits a warning in scan_rag"}
## Another heading, should be ok

The previous two headings were one after the other without blank lines in between. This text is followed by a heading without a blank separation. Hence...
## This should not be recognized as a heading.
And this text follows the false heading.

--- # this metadata block has an id and questions
~textid: cmmnt34
~questions: lines in the first block
---
This is some text immediately following a metadata block.

## Heading second level {class = heading}

This is text following a heading with properties. 

--- # metadata block with blank line in-between
~questions:
  - What are observational studies?
  - What are the differences between observational and experimental studies?
  - Interpretation of associations in observational studies
  - What do we mean with the term 'effects' in linear models?
  
~summary: |
  Describes the 'difference' between: modelling observational and experimental data,
  and the issues that "arise" when interpreting the output of linear models in these
  two conditions.
  We would also like to add another few notes.
---

####### This is text with 7 '#', should be recognized as simple text, prededed by metadata.

---  # isolated metadata with list
- start: the start
- second: 2
  another: item
---

In the following, we use an alternative coding of multiline strings.
---
~summary: >
  Describes the 'difference' between: modelling observational and experimental data,
  and the issues that "arise" when interpreting the output of linear models in these
  two conditions.
  We would also like to add another few notes.
---

There follows a non-conformant dictionary.

--- # non-conformant dictionary
(0, 1): a tuple
(1, 2): another tuple
---
text after non-conformant

--- # dictionary as value
first: 1
second:
  word: my word
  number: 1
...
text after nested dictionary.

--- # an error-raising dictionary
- start: a first line of a list
end: a tag
...
Text following error-raising dictionary

--- # a dictionary with a literal
literal string
...

--- # a dictionary with two literals
first literal
second literal
...

--- # a dictionary with a list
- first element
- second element
- 3
---

# Last first-level heading

A few words here to reach the min word count.

## Last second-level heading {comment = "not enough words"}
This is the final block, without blank line.