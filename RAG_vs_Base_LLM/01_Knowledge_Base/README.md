# Knowledge Base Specification & Provenance

## 1. Source Provenance
- **Repository**: LangChain Documentation (`langchain-ai/docs`)
- **Source URL**: `https://github.com/langchain-ai/docs`
- **Acquisition Date**: October 2026
- **Format**: Converted from Mintlify `.mdx` to standard UTF-8 Markdown (`.md`)

## 2. Scope & Selection Rationale
Rather than blindly indexing the full repository (which contains hundreds of enterprise provider integrations like Databricks, BigQuery, and Neo4j), this knowledge base isolates the core architectural primitives of the LangChain framework. This ensures:
1. Controlled corpus size (~10–20 documents, ~150–250 chunks).
2. Unambiguous ground truth for benchmark queries.
3. Defensible unanswerable question testing (testing absence of third-party modules).

## 3. Selected Documents
The following 10 files constitute the experimental knowledge base:

| Cleaned File | Original MDX Source | Topics & Concepts Covered |
| :--- | :--- | :--- |
| `01_models.md` | `src/oss/langchain/models.mdx` | Chat models, standalone vs. agent invocation, parameters (`max_retries`, `max_tokens`, `api_key`). |
| `02_retrieval.md` | `src/oss/deepagents/retrieval.mdx` | 2-step RAG architecture, knowledge base definition, indexing concepts. |
| `03_knowledge-base.md` | `src/oss/langchain/knowledge-base.mdx` | Vector stores, embeddings integration, semantic search workflow. |
| `04_messages.md` | `src/oss/langchain/messages.mdx` | `HumanMessage`, `SystemMessage`, `AIMessage`, chat message structures. |
| `05_context-engineering.md` | `src/oss/langchain/context-engineering.mdx` | Prompt assembly, context stuffing, token boundary management. |
| `06_quickstart.md` | `src/oss/langchain/quickstart.mdx` | End-to-end basic pipeline, invocation syntax, setup patterns. |
| `07_recursive_text_splitter.md` | `src/oss/python/splitters/recursive_text_splitter.mdx` | Default text chunking logic, separator hierarchies, `chunk_size`, `chunk_overlap`. |
| `08_character_text_splitter.md` | `src/oss/python/splitters/character_text_splitter.mdx` | Character-based splitting boundaries and fallbacks. |
| `09_markdown_header_metadata_splitter.md` | `src/oss/python/splitters/markdown_header_metadata_splitter.mdx` | Structural Markdown splitting, preserving header hierarchies as metadata. |
| `10_time-weighted-retriever.md` | `src/oss/python/retrievers/time-weighted-retriever.mdx` | Retriever decay scoring, combining recency with semantic similarity. |

## 4. Cleaning & Transformation Rules Applied
1. **Metadata Stripping**: Frontmatter YAML blocks (`---`) removed; `title` preserved as Markdown H1 (`# Title`).
2. **Component Removal**: React/JSX wrappers (`<Card>`, `<Tabs>`, `<Tip>`, `<Warning>`, `<Note>`) removed; text inside `<Note>` converted to Markdown blockquotes (`>`).
3. **Parameter Extraction**: `<ParamField>` converted into standard Markdown bullet points with types and default values.
4. **Language Deduplication**: JavaScript/TypeScript code tabs and `:::js` blocks purged; Python code fences preserved.
5. **Encoding Normalization**: Fixed Mojibake/UTF-8 artifacts (`â€™` to `'`, `â€œ` to `"`).