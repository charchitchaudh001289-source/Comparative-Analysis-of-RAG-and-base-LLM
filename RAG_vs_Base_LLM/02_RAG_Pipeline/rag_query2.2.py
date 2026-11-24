import time
import random
import json
from pathlib import Path

import chromadb
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError


# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================
# This section contains all settings required for the experiment.
# Keeping paths and model settings here makes them easy to change.

BASE_DIR = Path(__file__).resolve().parent          # 02_RAG_Pipeline/
PROJECT_ROOT = BASE_DIR.parent                     # RAG_vs_Base_LLM/

DB_DIR = str(BASE_DIR / "chroma_db")

# Change the filename when running the complete benchmark.
DATASET_PATH = PROJECT_ROOT / "03_BenchmarkQuestions" / "Qcontributors" / "final_question.json"

# Output file containing the combined experimental results.
OUTPUT_PATH = PROJECT_ROOT / "final_queryRun1.json"

MODEL_NAME = "gemini-3.5-flash-lite"

# Number of documents retrieved for each RAG question.
TOP_K = 3

# Maximum number of attempts for a failed Gemini API request.
MAX_RETRIES = 5

# Pause between API calls to stay under the 15 RPM Free Tier limit.
API_PAUSE = 4.5

# Instructions for the RAG model.
# The model must answer using only the retrieved documentation.
STRICT_RAG_INSTRUCTION = (
    "Answer using only the provided documentation. "
    "Do not use outside knowledge or access the internet. "
    "If the context does not contain sufficient information, "
    "state exactly: 'The provided context does not contain "
    "sufficient information to answer this question.'"
)


# ============================================================
# SECTION 2: INITIALIZATION
# ============================================================
# Connect to the existing ChromaDB database and Gemini API.

# Initialize ChromaDB.
chroma_client = chromadb.PersistentClient(path=DB_DIR)

collection = chroma_client.get_collection(
    name="langchain_kb"
)

print("Connected to ChromaDB.")
print("Collection count:", collection.count())
print("Database path:", DB_DIR)

# Initialize Gemini.
# The API key must be available through the environment.
gemini_client = genai.Client()


# ============================================================
# SECTION 3: GEMINI API CALL WITH RETRY
# ============================================================
# This function sends requests to Gemini.
# If the API temporarily fails because of rate limits (429)
# it waits 65 seconds for the quota window to reset.
# For server errors (503), it uses exponential backoff.

def call_gemini_with_retry(prompt, system_instruction):

    for attempt in range(MAX_RETRIES):

        try:
            response = gemini_client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    system_instruction=system_instruction
                )
            )

            return response

        except (ServerError, ClientError) as e:

            code = getattr(e, "code", None)
            error_message = str(e).lower()

            is_rate_limit = (
                code == 429
                or "resourceexhausted" in error_message
                or "429" in error_message
            )

            temporary_error = (
                is_rate_limit
                or code == 503
                or "unavailable" in error_message
                or "503" in error_message
            )

            # Do not retry errors that are not temporary.
            if not temporary_error:
                raise

            # Stop after the final unsuccessful attempt.
            if attempt == MAX_RETRIES - 1:
                raise

            if is_rate_limit:
                wait_time = 65
                print(
                    f"[Warning] Rate limit hit (429). "
                    f"Waiting {wait_time} seconds "
                    f"before retry {attempt + 1}/{MAX_RETRIES}..."
                )
            else:
                wait_time = (
                    (2 ** attempt) * 1.5
                    + random.uniform(0.5, 1.5)
                )
                print(
                    f"[Warning] API busy (503). "
                    f"Waiting {wait_time:.2f} seconds "
                    f"before retry {attempt + 1}/{MAX_RETRIES}..."
                )

            time.sleep(wait_time)


# ============================================================
# SECTION 4: TOKEN USAGE HELPER
# ============================================================
# Extract token counts from Gemini's response.
# This also handles cases where usage metadata is unavailable.

def get_token_usage(response):

    usage = getattr(response, "usage_metadata", None)

    return {
        "prompt_tokens": getattr(
            usage, "prompt_token_count", None
        ),
        "completion_tokens": getattr(
            usage, "candidates_token_count", None
        )
    }


# ============================================================
# SECTION 5: BASE LLM
# ============================================================
# Sends the original question directly to Gemini.
# No documentation is retrieved from ChromaDB.

def ask_base_llm(question):

    start_time = time.perf_counter()

    response = call_gemini_with_retry(
        prompt=question,
        system_instruction=(
            "You are a helpful and accurate "
            "technical documentation assistant."
        )
    )

    latency = time.perf_counter() - start_time

    token_usage = get_token_usage(response)

    return {
        "answer": response.text.strip() if response.text else "",
        "latency_sec": round(latency, 3),
        **token_usage
    }


# ============================================================
# SECTION 6: RAG LLM
# ============================================================
# Performs the complete RAG process:
# 1. Retrieve relevant documentation from ChromaDB.
# 2. Extract retrieved chunks, sources, and distances.
# 3. Add retrieved documentation to the prompt.
# 4. Send the prompt to Gemini.
# 5. Record generation time and token usage.

def ask_rag_llm(question, top_k=TOP_K):

    # -------------------- STEP 1: RETRIEVAL --------------------

    start_time = time.perf_counter()

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    total_chunks = collection.count()

    if total_chunks == 0:
        raise ValueError("ChromaDB collection is empty.")

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, total_chunks)
    )

    retrieval_time = time.perf_counter() - start_time

    # Extract retrieved documents.
    retrieved_chunks = (
        results["documents"][0]
        if results.get("documents") and results["documents"][0]
        else []
    )

    # Extract metadata associated with retrieved documents.
    metadata_list = (
        results["metadatas"][0]
        if results.get("metadatas") and results["metadatas"][0]
        else []
    )

    # Extract retrieval distances.
    distances = (
        results["distances"][0]
        if results.get("distances") and results["distances"][0]
        else []
    )

    # Identify the source document for each retrieved chunk.
    sources = [
        metadata.get("source", "unknown") if metadata else "unknown"
        for metadata in metadata_list
    ]

    # -------------------- STEP 2: PREPARE PROMPT --------------------

    context = "\n\n---\n\n".join(retrieved_chunks)

    user_prompt = (
        f"Context from documentation:\n"
        f"\"\"\"\n{context}\n\"\"\"\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    # -------------------- STEP 3: GENERATION --------------------

    start_time = time.perf_counter()

    response = call_gemini_with_retry(
        prompt=user_prompt,
        system_instruction=STRICT_RAG_INSTRUCTION
    )

    generation_time = time.perf_counter() - start_time

    token_usage = get_token_usage(response)

    # -------------------- STEP 4: RETURN RESULTS --------------------

    return {
        "answer": response.text.strip() if response.text else "",
        "retrieval_sec": round(retrieval_time, 3),
        "generation_sec": round(generation_time, 3),
        "total_latency_sec": round(
            retrieval_time + generation_time, 3
        ),
        **token_usage,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "distances": distances
    }


# ============================================================
# SECTION 7: LOAD BENCHMARK QUESTIONS
# ============================================================

def load_questions():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("Benchmark JSON must contain a list of questions.")

    required_fields = ["id", "question", "ground_truth"]

    for item in questions:
        for field in required_fields:
            if field not in item:
                raise ValueError(
                    f"Question is missing required field: {field}"
                )

    return questions


# ============================================================
# SECTION 8: SAVE EXPERIMENTAL RESULTS
# ============================================================
# Called after every question so completed results are preserved
# if a later API call fails or hits a daily quota cap.

def save_results(results):

    output_file = Path(OUTPUT_PATH)

    # Create the output directory if it does not exist.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


# ============================================================
# SECTION 9: MAIN EXPERIMENT EXECUTION
# ============================================================

if __name__ == "__main__":

    questions = load_questions()

    print("\n========================================")
    print("STARTING BASE LLM VS RAG EXPERIMENT")
    print("========================================")

    print("Total questions:", len(questions))
    print("Model:", MODEL_NAME)
    print("RAG top_k:", TOP_K)
    print("Output:", OUTPUT_PATH)

    # Auto-resume: load already-completed questions if output file exists.
    output_file = Path(OUTPUT_PATH)
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as file:
            all_results = json.load(file)
        completed_ids = {record["id"] for record in all_results}
        print(f"\n[Resume] Loaded {len(completed_ids)} already-completed questions.")
    else:
        all_results = []
        completed_ids = set()

    for item in questions:

        qid = item["id"]

        if qid in completed_ids:
            print(f"Skipping {qid} (already completed).")
            continue

        category = item.get("category", "uncategorized")
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n========== {qid} ==========")
        print("Category:", category)
        print("Question:", question)

        # -------------------- BASE LLM --------------------

        print("\n--- BASE LLM ---")

        base_output = ask_base_llm(question)

        print("Answer:", base_output["answer"])
        print("Latency:", base_output["latency_sec"], "seconds")
        print(
            "Tokens:",
            "Input =", base_output["prompt_tokens"],
            "| Output =", base_output["completion_tokens"]
        )

        time.sleep(API_PAUSE)

        # -------------------- RAG LLM --------------------

        print("\n--- RAG LLM ---")

        rag_output = ask_rag_llm(question, top_k=TOP_K)

        print("Answer:", rag_output["answer"])
        print("Sources:", rag_output["sources"])

        print(
            "Retrieval:", rag_output["retrieval_sec"], "seconds",
            "| Generation:", rag_output["generation_sec"], "seconds",
            "| Total:", rag_output["total_latency_sec"], "seconds"
        )

        print(
            "Tokens:",
            "Input =", rag_output["prompt_tokens"],
            "| Output =", rag_output["completion_tokens"]
        )

        # -------------------- COMBINE RESULTS --------------------

        record = {
            "id": qid,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "base": base_output,
            "rag": rag_output
        }

        all_results.append(record)
        completed_ids.add(qid)

        # Save results immediately after completing each question.
        save_results(all_results)

        print("\nResults saved for", qid)
        print("-" * 50)

        time.sleep(API_PAUSE)

    # -------------------- EXPERIMENT COMPLETE --------------------

    print("\n========================================")
    print("EXPERIMENT COMPLETED")
    print("========================================")

    print("Total questions processed:", len(all_results))
    print("Results saved to:", OUTPUT_PATH)