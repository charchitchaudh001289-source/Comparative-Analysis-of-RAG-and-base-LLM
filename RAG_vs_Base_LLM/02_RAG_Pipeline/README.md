# README – Base LLM vs. RAG Pipeline

## 1. Project Overview

This Python script implements two approaches to answering questions about LangChain technical documentation:

1. **Base LLM (Large Language Model):** Sends the user's question directly to Google's Gemini model without retrieving supporting documents from the local knowledge base.
2. **RAG (Retrieval-Augmented Generation):** Searches a locally stored ChromaDB vector database for relevant documentation chunks, combines the retrieved information into a context, and sends that context along with the question to Gemini.

Both approaches use the same Gemini model, `gemini-3.5-flash-lite`, allowing us to compare their responses, latency, and token consumption.

The primary purpose is to evaluate whether providing relevant documentation through retrieval improves the accuracy and reliability of an LLM's answers.

---

## 2. Libraries and Imports

The script uses several Python libraries and external packages.

```python
import os
import time
import random
import chromadb
import json
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
```

### Explanation of each import

| Import | Purpose |
|---|---|
| `os` | Provides operating-system-related functionality. Imported but currently not used in this script. |
| `time` | Measures execution time and introduces delays between API requests. |
| `random` | Generates random delays for retry attempts to prevent repeated API requests from occurring at predictable intervals. |
| `chromadb` | Connects to and queries the local vector database containing LangChain documentation. |
| `json` | Reads benchmark questions from a JSON file. |
| `Path` | Provides a convenient way to work with file paths. |
| `genai` | Provides access to Google's Gemini API. |
| `types` | Provides configuration classes for Gemini API requests. |
| `ServerError`, `ClientError` | Used to identify and handle certain API errors. |

---

## 3. Database Initialization

```python
DB_DIR = r"C:\Users\Nathan\Videos\AI Project\RAG_vs_Base_LLM\02_RAG_Pipeline\chroma_db"

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(name="langchain_kb")
```

This section establishes a connection to the locally stored ChromaDB database.

**`DB_DIR`**

Specifies the directory where the persistent ChromaDB database is stored.

The `r` before the path creates a raw Python string, preventing backslashes from being interpreted as escape characters.

**`chromadb.PersistentClient(path=DB_DIR)`**

Creates a connection to the persistent ChromaDB database.

Unlike an in-memory database, a persistent database stores its data on disk, allowing the application to reuse previously indexed documentation without rebuilding the database every time.

**`get_collection(name="langchain_kb")`**

Retrieves the existing collection named `langchain_kb`.

A ChromaDB collection stores document chunks, their vector embeddings, and associated metadata. These embeddings allow the database to find documentation that is semantically similar to a user's question.

The database and collection must already exist before this script runs.

---

## 4. Gemini API Initialization

```python
gemini_client = genai.Client()
MODEL_NAME = "gemini-3.5-flash-lite"
```

This section initializes the Gemini API client.

- `genai.Client()` creates a client used to communicate with Google's Gemini API.
- `MODEL_NAME` specifies which Gemini model will generate the responses.

The API client obtains its credentials from the configured environment or Google GenAI SDK authentication settings.

Both the Base LLM and RAG implementations use this same model.

---

## 5. RAG System Instruction

```python
STRICT_RAG_SYSTEM_INSTRUCTION = (
    "You are an authoritative technical assistant evaluating LangChain documentation. "
    "Answer the user's question solely and strictly using the provided context. "
    "If the provided context does not contain sufficient information to answer the question with certainty, "
    "you MUST state exactly: 'The provided context does not contain sufficient information to answer this question.' "
    "Do not extrapolate, infer, or use outside knowledge."
)
```

This variable defines the system-level instructions given to Gemini when generating RAG responses.

Its purpose is to restrict the model to the information retrieved from the LangChain documentation.

It establishes three main rules:

1. The answer must be based exclusively on the supplied context.
2. If the context does not contain sufficient information, the model must return a predefined refusal statement.
3. The model must not use external knowledge or make unsupported assumptions.

This instruction is particularly important when evaluating RAG because it helps distinguish answers supported by retrieved documents from answers generated using the model's general knowledge.

**Note:** The Base LLM does not receive this strict instruction. It receives a separate general-purpose system instruction.

---

## 6. API Retry Mechanism

```python
def call_gemini_with_retry(prompt: str, system_instruction: str, max_retries: int = 5):
```

This function sends requests to Gemini and handles temporary API failures.

API services can occasionally return errors because of rate limits, temporary server problems, or service unavailability.

Rather than immediately terminating the program, this function attempts to retry certain requests.

### 6.1 Sending the API request

```python
return gemini_client.models.generate_content(
    model=MODEL_NAME,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0.0,
        system_instruction=system_instruction
    )
)
```

This sends a content-generation request to Gemini.

- `model`: Specifies the Gemini model.
- `contents`: Contains the actual prompt sent to the model.
- `temperature=0.0`: Reduces randomness in the generated responses.
- `system_instruction`: Provides instructions that guide the model's behaviour.

Using a temperature of zero aims to make responses more consistent across repeated runs, although it does not guarantee identical outputs.

### 6.2 Handling API errors

```python
except (ServerError, ClientError) as e:
    code = getattr(e, "code", None)
```

This catches certain server-side and client-side API exceptions and attempts to retrieve their error code.

```python
if code in [429, 503] or "ResourceExhausted" in str(e) or "UNAVAILABLE" in str(e):
```

The function checks for:

- `429`: Too Many Requests or rate-limit-related errors.
- `503`: Service unavailable.
- `ResourceExhausted`: Indicates that a resource or quota limit has been reached.
- `UNAVAILABLE`: Indicates that the service is temporarily unavailable.

Other errors are raised instead of being retried automatically.

### 6.3 Exponential backoff

```python
wait_time = (2 ** attempt) * 1.5 + random.uniform(0.5, 1.5)
time.sleep(wait_time)
```

This implements exponential backoff with random jitter.

The waiting time increases after each failed attempt, reducing the frequency of repeated API requests.

The random component helps prevent multiple requests from retrying simultaneously.

The function allows a maximum of five attempts by default.

**Why this matters:** It makes the experiment more resilient to temporary API failures and reduces the likelihood that one temporary service interruption will stop the entire benchmark.

---

## 7. Base LLM Implementation

```python
def ask_base_llm(question: str) -> dict:
```

This function sends a benchmark question directly to Gemini without retrieving any documents from ChromaDB.

It represents the baseline system used in the experiment.

### 7.1 Measuring the execution time

```python
t0 = time.perf_counter()
```

`time.perf_counter()` records a high-resolution timestamp.

It is used to measure how long the API request takes.

### 7.2 Calling Gemini

```python
response = call_gemini_with_retry(
    prompt=question,
    system_instruction="You are a helpful and accurate technical documentation assistant."
)
```

The original question is sent directly to Gemini.

There is no document retrieval, vector search, or additional documentation context.

The model generates its response using its existing knowledge and the instructions provided.

### 7.3 Calculating latency

```python
latency_sec = time.perf_counter() - t0
```

Calculates the elapsed time between the beginning of the request and the receipt of the response.

This measurement includes the API call and any retry waiting time.

### 7.4 Returning the results

```python
return {
    "answer": response.text.strip() if response.text else "",
    "latency_sec": round(latency_sec, 3),
    "prompt_tokens": response.usage_metadata.prompt_token_count,
    "completion_tokens": response.usage_metadata.candidates_token_count
}
```

The function returns four values:

| Field | Meaning |
|---|---|
| `answer` | The response generated by Gemini. |
| `latency_sec` | Total time taken to generate the response. |
| `prompt_tokens` | Number of tokens used in the input prompt. |
| `completion_tokens` | Number of tokens generated in the response. |

These measurements are later used to compare the Base LLM against the RAG implementation.

---

## 8. RAG Implementation

```python
def ask_rag_llm(question: str, top_k: int = 10) -> dict:
```

This function implements the Retrieval-Augmented Generation pipeline.

Unlike the Base LLM, this approach retrieves relevant documentation from ChromaDB before generating an answer.

The function has two major stages:

1. Retrieval
2. Generation

### 8.1 Retrieval stage

```python
t_retrieval_0 = time.perf_counter()

results = collection.query(
    query_texts=[question],
    n_results=top_k
)

retrieval_sec = time.perf_counter() - t_retrieval_0
```

The user's question is submitted to the ChromaDB collection.

ChromaDB compares the question's embedding against stored document embeddings and retrieves the most semantically similar document chunks.

**Understanding `top_k`:**

`top_k` determines how many document chunks ChromaDB should retrieve.

For example:

- `top_k=3`: Retrieves three chunks.
- `top_k=5`: Retrieves five chunks.
- `top_k=10`: Retrieves ten chunks.

The current implementation uses `top_k=10`.

Increasing this value provides the model with more potentially relevant information. However, retrieving more chunks does not necessarily improve answer quality. It can also introduce irrelevant context and increase token consumption.

### 8.2 Extracting retrieved documents

```python
retrieved_chunks = results["documents"][0] if results["documents"] else []
metadata_list = results["metadatas"][0] if results["metadatas"] else []
distances = results["distances"][0] if results.get("distances") else []

sources = [m.get("source", "unknown") for m in metadata_list]
```

This section extracts information from the ChromaDB search results.

**`retrieved_chunks`**

Contains the actual text content of the retrieved documentation chunks.

These chunks will eventually become the context provided to Gemini.

**`metadata_list`**

Contains metadata associated with each retrieved chunk, such as its source document.

**`distances`**

Contains the distance values returned by ChromaDB for the retrieved results.

These values help inspect how closely the retrieved documents match the search query. Their interpretation depends on the distance metric configured for the collection.

**`sources`**

Extracts the source document names from the metadata.

This allows the program to report which documentation files contributed to the retrieved context.

### 8.3 Debugging retrieved chunks

The script currently contains a debugging section that has been commented out using triple quotation marks.

```python
''' print("\n--- RETRIEVED CHUNKS (DEBUG) ---")

for i, chunk in enumerate(retrieved_chunks):
    source = metadata_list[i].get("source", "unknown")
    print(f"\nChunk {i + 1}")
    print(f"Source: {source}")

    if i < len(distances):
        print(f"Distance: {distances[i]:.4f}")

    print("Content:")
    print(chunk[:1500])
    print("-" * 40)'''
```

When activated, this section prints:

- The retrieved chunk number.
- The source document.
- The distance value.
- The first 1,500 characters of the retrieved text.

This is useful for investigating retrieval failures.

For example, if the correct information exists in the database but is not included in the retrieved chunks, the problem may be related to semantic search, chunking, or retrieval ranking.

If the correct information is retrieved but Gemini still refuses to answer, the problem may instead involve prompt construction or response generation.

The debugging section is currently disabled so that the benchmark output remains shorter and easier to read.

### 8.4 Constructing the context

```python
context_text = "\n\n---\n\n".join(retrieved_chunks)
```

This combines all retrieved document chunks into a single string.

A separator (`---`) is inserted between chunks to make the boundaries between retrieved documents clearer.

For example, if three chunks are retrieved, they become one combined context containing all three pieces of documentation.

### 8.5 Constructing the RAG prompt

```python
user_prompt = (
    f"Context from documentation:\n"
    f"\"\"\"\n{context_text}\n\"\"\"\n\n"
    f"Question: {question}\n\n"
    f"Answer:"
)
```

This creates the final prompt sent to Gemini.

It contains two important components:

1. **Context:** The documentation chunks retrieved from ChromaDB.
2. **Question:** The original benchmark question.

The context is enclosed in quotation marks using triple quotation marks to distinguish it from the rest of the prompt.

The model is then instructed to answer the question using the supplied context.

### 8.6 Generation stage

```python
t_gen_0 = time.perf_counter()

response = call_gemini_with_retry(
    prompt=user_prompt,
    system_instruction=STRICT_RAG_SYSTEM_INSTRUCTION
)

generation_sec = time.perf_counter() - t_gen_0
```

The constructed prompt is sent to Gemini.

Unlike the Base LLM, the RAG implementation uses the strict system instruction defined earlier.

The generation timer measures how long Gemini takes to produce its answer, including any retry delays.

### 8.7 Returning RAG results

```python
return {
    "answer": response.text.strip() if response.text else "",
    "retrieval_sec": round(retrieval_sec, 3),
    "generation_sec": round(generation_sec, 3),
    "total_latency_sec": round(retrieval_sec + generation_sec, 3),
    "prompt_tokens": response.usage_metadata.prompt_token_count,
    "completion_tokens": response.usage_metadata.candidates_token_count,
    "sources": sources
}
```

The RAG function returns the following information:

| Field | Meaning |
|---|---|
| `answer` | Answer generated using retrieved documentation. |
| `retrieval_sec` | Time required to search ChromaDB. |
| `generation_sec` | Time required for Gemini to generate the answer. |
| `total_latency_sec` | Combined retrieval and generation time. |
| `prompt_tokens` | Tokens used in the complete RAG prompt. |
| `completion_tokens` | Tokens generated in the response. |
| `sources` | Source documents associated with the retrieved chunks. |

One important distinction is that RAG prompt tokens include the retrieved documentation context, whereas Base LLM prompt tokens generally contain only the question and system instruction.

---

## 9. Loading Benchmark Questions

```python
def load_questions():
    file_path = Path(r"C:\Users\Nathan\Videos\AI Project\RAG_vs_Base_LLM\03_BenchmarkQuestions\benchmarkq2.json")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
```

This function loads benchmark questions from an external JSON file.

Instead of manually writing each question into the Python script, the program reads the questions from `benchmarkq2.json`.

The JSON file contains structured benchmark information, such as:

- Question ID
- Question category
- Question text
- Ground-truth answer
- Evidence passage
- Source document

The `json.load()` function converts the JSON data into Python objects, allowing the program to iterate through the questions.

This separation makes it easier to modify or expand the benchmark dataset without changing the main Python implementation.

---

## 10. Main Execution Block

```python
if __name__ == "__main__":
```

This condition ensures that the benchmark execution code runs when the Python file is executed directly.

It prevents the main benchmark loop from automatically running if the script is imported as a module by another Python file.

### 10.1 Loading questions

```python
questions = load_questions()
```

Loads the benchmark questions from the JSON file.

### 10.2 Iterating through questions

```python
for item in questions:
    question = item["question"]
    print(f"\nTesting Question: {question}\n")
```

The program iterates through each benchmark entry.

For every entry, it extracts the question text and displays it in the terminal.

### 10.3 Running the Base LLM

```python
print("--- [1] BASE LLM ---")

base_output = ask_base_llm(question)

print(f"Answer: {base_output['answer']}")
```

The question is sent to the Base LLM implementation.

The program prints the generated answer.

It also displays the latency and token consumption:

```python
print(
    f"Latency: {base_output['latency_sec']}s | "
    f"Tokens: in={base_output['prompt_tokens']}, "
    f"out={base_output['completion_tokens']}\n"
)
```

### 10.4 Introducing a delay

```python
time.sleep(2.0)
```

Pauses the program for two seconds before making the next API request.

This helps space out requests to the Gemini API and reduces the likelihood of sending requests too frequently.

### 10.5 Running the RAG implementation

```python
print("--- [2] RAG LLM ---")

rag_output = ask_rag_llm(question, top_k=10)
```

The same question is passed to the RAG implementation.

The function retrieves ten relevant chunks from ChromaDB and sends the combined context and question to Gemini.

The program then displays the generated answer and its sources:

```python
print(f"Answer: {rag_output['answer']}")
print(f"Sources: {rag_output['sources']}")
```

It also displays the retrieval time, generation time, total latency, and token consumption.

```python
print(
    f"Retrieval: {rag_output['retrieval_sec']}s | "
    f"Generation: {rag_output['generation_sec']}s | "
    f"Total: {rag_output['total_latency_sec']}s"
)

print(
    f"Tokens: in={rag_output['prompt_tokens']}, "
    f"out={rag_output['completion_tokens']}"
)
```

Finally, a separator is printed before moving to the next benchmark question.

---

## 11. Overall Execution Workflow

The complete execution process can be summarized as follows:

1. Import required libraries.
2. Connect to the existing ChromaDB collection.
3. Initialize the Gemini API client.
4. Load benchmark questions from the JSON file.
5. Select one question from the benchmark.
6. Send the question directly to Gemini using the Base LLM function.
7. Record the Base LLM answer, latency, and token consumption.
8. Retrieve the most relevant documentation chunks from ChromaDB.
9. Combine the retrieved chunks into a context.
10. Send the context and original question to Gemini using the RAG function.
11. Record the RAG answer, retrieval time, generation time, total latency, token consumption, and sources.
12. Repeat the process for the remaining benchmark questions.

### Simplified workflow diagram

```text
             Benchmark JSON
                   |
                   v
             Load Questions
                   |
                   v
             Select Question
                   |
           +-------+-------+
           |               |
           v               v
       BASE LLM          RAG SYSTEM
           |               |
           |               v
           |         ChromaDB Search
           |               |
           |               v
           |         Retrieve Top-K
           |         Document Chunks
           |               |
           |               v
           |         Combine Context
           |               |
           |               v
           |         Gemini + Context
           |               |
           v               v
       Base Answer      RAG Answer
           |               |
           v               v
       Record Metrics   Record Metrics
           |               |
           +-------+-------+
                   |
                   v
             Next Question
```

---

## 12. Important Experimental Considerations

### Same generation model

Both implementations use the same Gemini model. This is important because the experiment aims to compare direct LLM answering against retrieval-augmented answering rather than comparing two different generation models.

### Different system instructions

The Base LLM receives a general technical assistant instruction, while RAG receives a strict context-only instruction.

This is an important experimental difference to document. The comparison therefore evaluates both the retrieval process and the associated prompting strategy, not retrieval alone.

### Temperature

Both implementations use `temperature=0.0` to reduce response randomness and improve consistency.

### Retrieval parameter (`top_k`)

The current implementation retrieves ten document chunks per question.

This value should be treated as an experimental setting. Different values, such as 3, 5, and 10, can be compared to investigate how retrieval quantity affects answer accuracy, latency, and token usage.

### Token consumption

RAG generally uses more input tokens because it includes retrieved documentation in its prompt.

This additional context is part of the retrieval-augmented approach and should be considered when evaluating operational cost.

### Ground-truth evaluation

The current script prints model answers and performance measurements but does not automatically calculate answer accuracy.

The benchmark JSON contains ground-truth answers and evidence passages that can be used in a separate evaluation stage.

---

## 13. Current Limitations and Future Improvements

The current implementation provides the basic functionality required to compare a Base LLM with a RAG-based system. Several improvements can be introduced later:

1. **Automated evaluation:** Compare generated answers against benchmark ground truth using exact match, token F1, semantic similarity, or an appropriate evaluator.
2. **Result storage:** Save outputs and measurements to JSON or CSV instead of relying only on terminal output.
3. **Retrieval evaluation:** Measure whether the correct evidence passage appears among the retrieved chunks.
4. **Top-k comparison:** Evaluate different retrieval values to determine their impact on accuracy, latency, and token consumption.
5. **Failure analysis:** Distinguish between retrieval failures, where relevant evidence is not retrieved, and generation failures, where relevant evidence is retrieved but not used correctly.
6. **Improved experiment controls:** Use comparable system instructions for both approaches if the goal is to isolate the effect of retrieval itself.

---

## 14. Summary

This script establishes two question-answering approaches using the same Gemini model.

The Base LLM answers questions directly using its existing model knowledge, whereas the RAG implementation first retrieves relevant information from a local ChromaDB knowledge base and supplies that information to Gemini as additional context.

By recording answers, latency, token consumption, and retrieved sources, the implementation provides the foundation for a comparative evaluation of the two approaches.

The next stage of the project is to evaluate the generated answers against the benchmark ground truth and analyze the differences in accuracy, retrieval effectiveness, latency, and operational cost.