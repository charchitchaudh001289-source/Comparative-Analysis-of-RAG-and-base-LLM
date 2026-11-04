---
title: "Splitting recursively - Text splitter integration guide"
---

This [text splitter](/oss/integrations/splitters/) is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is `["\n\n", "\n", " ", ""]`. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

1. How the text is split: by list of characters.
2. How the chunk size is measured: by number of characters.

Below we show example usage.

```shell
pip install -qU langchain-text-splitters
```

To obtain the string content directly, use `.split_text`.

To create LangChain @[Document] objects (e.g., for use in downstream tasks), use `.create_documents`.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load example document
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.create_documents([state_of_the_union])
print(texts[0])
print(texts[1])
```
```python
page_content='Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and'
page_content='of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.'
```

```python
print(text_splitter.split_text(state_of_the_union)[:2])
```

```python
['Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and',
 'of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.']
```

- `chunk_size`: The maximum size of a chunk, where size is determined by the `length_function`.
- `chunk_overlap`: Target overlap between chunks. Overlapping chunks helps to mitigate loss of information when context is divided between chunks.
- `length_function`: Function determining the chunk size.
- `is_separator_regex`: Whether the separator list (defaulting to `["\n\n", "\n", " ", ""]`) should be interpreted as regex.

```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    # Existing args
)
```
