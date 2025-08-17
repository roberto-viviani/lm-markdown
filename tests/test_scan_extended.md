--- # valid header
title: testing invalid metadata blocks and headings
---

To use this document for testing:
- to test parse_markdown, load the document with load_blocks and verify the error blocks:
    print(blocklist_get_info(blocks))
- to test inline feedback, scan the document:
    python -m library.scan test/test_md_heads.md test_md.md

We start with a valid metadata block. This should produce no errors.

---
~questions: Is this a valid metadata block?
---

Here, we produce a header that triggers a YAML parse error.

---
- a first list element
next: no longer a list
---

The following is a valid YAML block, but not usable for LM markdown as it only contains literals.

---
this is first line
a second line
---

The following is an empty metadata block

---
---

The following is metadata containing a list of literals

---
- first
- second
---

Another case, a list of a dict and another nested list

---
-
    first: 1
    second: 2
-   
    - one
    - two
---

The following contains a dictionary with int as key

---
1: the number one
...

The following is a list of dictionaries, with the first dictionary having
an int as key

---
- 1: the number one
- test: the test line
...

This will be parsed correctly:

---
- test1: the first line
- test2: the second line
...

# This is a valid heading   {class = any}

The following is a empty heading, not followed by space, categorized as text.

##

The following has a space and is categorized as empty heading

## 

A empty heading with data

## {class = any}

Now, we create a metadata blocks, but we do not close it validly.

---
~questions: I do not know
--

Now, some text that will create an unclosed metadata block error, unless a new metadata block is defined after this one, which will produce parse errors (also possible otherwise, depending on what follows the metadata).


