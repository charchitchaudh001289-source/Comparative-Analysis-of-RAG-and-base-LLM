import os
import time
import chromadb
from google import genai
from google.genai import types

# 1. Setup paths and client instances
DB_DIR = r"C:\Users\Nathan\Videos\AI Project\RAG_vs_Base_LLM\02_RAG_Pipeline\chroma_db"

# Initialize ChromaDB persistent client and load the existing collection
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(name="langchain_kb")

# Initialize Gemini client (picks up GEMINI_API_KEY from environment)
gemini_client = genai.Client()

# Constant model identifier (can be swapped later, e.g. to gemini-2.0-flash)
MODEL_NAME = "gemini-3.5-flash-lite"

# System instruction to enforce factual refusal and prevent hallucinations
STRICT_RAG_SYSTEM_INSTRUCTION = (
    "You are an authoritative technical assistant evaluating LangChain documentation. "
    "Answer the user's question solely and strictly using the provided context. "
    "If the provided context does not contain sufficient information to answer the question with certainty, "
    "you MUST state exactly: 'The provided context does not contain sufficient information to answer this question.' "
    "Do not extrapolate, infer, or use outside knowledge."
)


def ask_base_llm(question: str) -> dict:
    """Queries Gemini directly without retrieval context."""
    t0 = time.perf_counter()

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=question,
        config=types.GenerateContentConfig(
            temperature=0.0,
            system_instruction="You are a helpful and accurate technical documentation assistant."
            #thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )

    latency_sec = time.perf_counter() - t0

    return {
        "answer": response.text.strip() if response.text else "",
        "latency_sec": round(latency_sec, 3),
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count
    }


def ask_rag_llm(question: str, top_k: int = 3) -> dict:
    """Retrieves relevant chunks from ChromaDB, constructs grounded context, and queries Gemini."""
    # Phase 1: Retrieval
    t_retrieval_0 = time.perf_counter()

    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    retrieval_sec = time.perf_counter() - t_retrieval_0

    # Extract text chunks and unique sources
    retrieved_chunks = results["documents"][0] if results["documents"] else []
    metadata_list = results["metadatas"][0] if results["metadatas"] else []
    sources = [m.get("source", "unknown") for m in metadata_list]

    # Format context block
    context_text = "\n\n---\n\n".join(retrieved_chunks)

    user_prompt = (
        f"Context from documentation:\n"
        f"\"\"\"\n{context_text}\n\"\"\"\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # Phase 2: Generation
    t_gen_0 = time.perf_counter()

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=STRICT_RAG_SYSTEM_INSTRUCTION,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )

    generation_sec = time.perf_counter() - t_gen_0
    total_latency_sec = retrieval_sec + generation_sec

    return {
        "answer": response.text.strip() if response.text else "",
        "retrieval_sec": round(retrieval_sec, 3),
        "generation_sec": round(generation_sec, 3),
        "total_latency_sec": round(total_latency_sec, 3),
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
        "sources": sources
    }


if __name__ == "__main__":
    # Smoke test comparison using a known parameter from models.md
    test_question = "What is the default value of the max_retries parameter in init_chat_model?"

    print(f"Testing Question: {test_question}\n")

    print("--- [1] BASE LLM ---")
    base_output = ask_base_llm(test_question)
    print(f"Answer: {base_output['answer']}")
    print(f"Latency: {base_output['latency_sec']}s | Tokens: in={base_output['prompt_tokens']}, out={base_output['completion_tokens']}\n")

    print("--- [2] RAG LLM ---")
    rag_output = ask_rag_llm(test_question, top_k=3)
    print(f"Answer: {rag_output['answer']}")
    print(f"Sources: {rag_output['sources']}")
    print(f"Retrieval: {rag_output['retrieval_sec']}s | Generation: {rag_output['generation_sec']}s | Total: {rag_output['total_latency_sec']}s")
    print(f"Tokens: in={rag_output['prompt_tokens']}, out={rag_output['completion_tokens']}")