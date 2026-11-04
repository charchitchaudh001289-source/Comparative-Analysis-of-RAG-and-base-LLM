import time
import random
import chromadb
import json
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

DB_DIR = r"C:\Users\Nathan\Videos\AI Project\RAG_vs_Base_LLM\02_RAG_Pipeline\chroma_db"

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_collection(name="langchain_kb")

# Connect to Gemini
gemini_client = genai.Client()
MODEL_NAME = "gemini-3.5-flash-lite"

STRICT_RAG_SYSTEM_INSTRUCTION = (
    "You are an authoritative technical assistant evaluating LangChain documentation. "
    "Answer the user's question solely and strictly using the provided context. "
    "If the provided context does not contain sufficient information to answer the question with certainty, "
    "you MUST state exactly: 'The provided context does not contain sufficient information to answer this question.' "
    "Do not extrapolate, infer, or use outside knowledge."
)


def call_gemini_with_retry(prompt: str, system_instruction: str, max_retries: int = 5):
    """Executes a Gemini call with exponential backoff on transient 429/503 errors."""
    for attempt in range(max_retries):
        try:
            return gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    system_instruction=system_instruction
                )
            )
        except (ServerError, ClientError) as e:
            code = getattr(e, "code", None)
            if code in [429, 503] or "ResourceExhausted" in str(e) or "UNAVAILABLE" in str(e):
                if attempt == max_retries - 1:
                    raise e
                wait_time = (2 ** attempt) * 1.5 + random.uniform(0.5, 1.5)
                print(f"[Warning] API rate-limit/busy ({code}). Waiting {round(wait_time, 2)}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise e


def ask_base_llm(question: str) -> dict:
    t0 = time.perf_counter()

    response = call_gemini_with_retry(
        prompt=question,
        system_instruction="You are a helpful and accurate technical documentation assistant."
    )

    latency_sec = time.perf_counter() - t0

    return {
        "answer": response.text.strip() if response.text else "",
        "latency_sec": round(latency_sec, 3),
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count
    }


def ask_rag_llm(question: str, top_k: int = 3) -> dict:
    # 1. Retrieval
    t_retrieval_0 = time.perf_counter()
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    retrieval_sec = time.perf_counter() - t_retrieval_0

    #retrieved_chunks = results["documents"][0] if results["documents"] else []
    #metadata_list = results["metadatas"][0] if results["metadatas"] else []
    #sources = [m.get("source", "unknown") for m in metadata_list]

    
    retrieved_chunks = results["documents"][0] if results["documents"] else []
    metadata_list = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results.get("distances") else []

    sources = [m.get("source", "unknown") for m in metadata_list]

# DEBUG: inspect what ChromaDB actually retrieved
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

    
    context_text = "\n\n---\n\n".join(retrieved_chunks)

    user_prompt = (
        f"Context from documentation:\n"
        f"\"\"\"\n{context_text}\n\"\"\"\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # 2. Generation (using retry wrapper)
    t_gen_0 = time.perf_counter()
    response = call_gemini_with_retry(
        prompt=user_prompt,
        system_instruction=STRICT_RAG_SYSTEM_INSTRUCTION
    )
    generation_sec = time.perf_counter() - t_gen_0

    return {
        "answer": response.text.strip() if response.text else "",
        "retrieval_sec": round(retrieval_sec, 3),
        "generation_sec": round(generation_sec, 3),
        "total_latency_sec": round(retrieval_sec + generation_sec, 3),
        "prompt_tokens": response.usage_metadata.prompt_token_count,
        "completion_tokens": response.usage_metadata.candidates_token_count,
        "sources": sources
    }

def load_questions():
    file_path = Path(r"C:\Users\Nathan\Videos\AI Project\RAG_vs_Base_LLM\03_BenchmarkQuestions\langchain_messages_eval_questions.json")
    with open(file_path,"r",encoding="utf-8") as file:
        return json.load(file)

if __name__ =="__main__":

    """results = collection.query(
        query_texts=[
            "Network errors, rate limits (429), and server errors (5xx) are retried automatically"
        ],
        n_results=5 )

    print("\n--- SEARCH TEST RESULTS ---")

    for doc in results["documents"][0]:
        print(doc)
        print("-" * 40)"""


    questions = load_questions()


    for item in questions:
        question = item["question"]
        print(f"\nTesting Question: {question}\n")

        print("--- [1] BASE LLM ---")
        base_output = ask_base_llm(question)
        print(f"Answer: {base_output['answer']}")
        print(
        f"Latency: {base_output['latency_sec']}s | "
        f"Tokens: in={base_output['prompt_tokens']}, "
        f"out={base_output['completion_tokens']}\n")


        time.sleep(2.0)

        print("--- [2] RAG LLM ---")
        rag_output = ask_rag_llm(question, top_k=3)
        print(f"Answer: {rag_output['answer']}")
        print(f"Sources: {rag_output['sources']}")
        print(
        f"Retrieval: {rag_output['retrieval_sec']}s | "
        f"Generation: {rag_output['generation_sec']}s | "
        f"Total: {rag_output['total_latency_sec']}s")
        print(
        f"Tokens: in={rag_output['prompt_tokens']}, "
        f"out={rag_output['completion_tokens']}")
        print("\n" + "-" * 50)
