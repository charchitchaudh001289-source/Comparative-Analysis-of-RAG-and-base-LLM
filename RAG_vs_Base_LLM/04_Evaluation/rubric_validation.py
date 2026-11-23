import json
import time
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import ServerError

# ------------------------------------------------------------
# CONFIGURATION (Matches evaluation_script2.py)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "rubric_validation_results.json"

JUDGE_MODEL = "gemini-3.1-flash-lite"
client = genai.Client()


def generate_judge_response(prompt: str):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
        except ServerError as error:
            error_code = getattr(error, "code", None)
            if error_code != 503 or attempt == max_retries - 1:
                raise
            wait_time = 2 ** (attempt + 1)
            print(f"Gemini temporarily unavailable (503). Retrying in {wait_time}s...")
            time.sleep(wait_time)


def evaluate_pair(question: str, ground_truth: str, answer_a: str, answer_b: str) -> dict:
    prompt = f"""
You are an impartial evaluator comparing two answers to a question.

Evaluate both answers using the supplied ground truth.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Evaluate each answer using these three criteria:

1. Factual Correctness:
   Does the answer provide factually correct information
   according to the ground truth?

2. Completeness:
   Does the answer include all the important information
   required to answer the question?

3. Relevance:
   Does the answer directly address the question without
   unnecessary or unrelated information?

Give each criterion a score from 1 to 5.

Scoring guide:
1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Use the ground truth as the reference.
Do not reward an answer for adding unsupported information.

Return your evaluation in this exact JSON structure:

{{
    "answer_a": {{
        "factual_correctness": 1,
        "completeness": 1,
        "relevance": 1
    }},
    "answer_b": {{
        "factual_correctness": 1,
        "completeness": 1,
        "relevance": 1
    }},
    "reasoning": "Briefly explain the evaluation."
}}

Return valid JSON only.
"""
    response = generate_judge_response(prompt)
    return json.loads(response.text)


CONTROLLED_TEST_CASES = [
    {
        "test_id": "TC01_Architecture_Ratings",
        "source_question_id": "Q101",
        "question": "According to the architecture comparison table in LangChain's retrieval guide, what are the explicit ratings for 2-Step RAG across developer control, flexibility, and latency, and what canonical production use cases are cited?",
        "ground_truth": "Developer Control: High. Flexibility: Low. Latency: Fast. Representative use cases: FAQs and documentation bots.",
        "answers": {
            "correct": "Developer Control is High, Flexibility is Low, and Latency is Fast. Canonical use cases are FAQs and documentation bots.",
            "partially_correct": "Developer Control is High and Latency is Fast, making it good for simple bots.",
            "incorrect": "Developer Control is Low, Flexibility is High, and Latency is Variable, suited for complex multi-tool research agents.",
            "correct_with_fluff": "LangChain was founded by Harrison Chase and supports Python and JavaScript integrations across OpenAI, Anthropic, and Google Gemini models, as well as vector stores like FAISS and Chroma. In the comparison table, 2-Step RAG has High developer control, Low flexibility, and Fast latency, and is used for FAQs and documentation bots."
        }
    },
    {
        "test_id": "TC02_Default_Separators",
        "source_question_id": "Q113",
        "question": "What is the exact default list of separators used by RecursiveCharacterTextSplitter, and in what order does it attempt to split the input text?",
        "ground_truth": "The default separator list is [\"\\n\\n\", \"\\n\", \" \", \"\"], splitting in order of double newlines, single newlines, spaces, and empty strings.",
        "answers": {
            "correct": "The default separator list is [\"\\n\\n\", \"\\n\", \" \", \"\"], splitting in order of double newlines, single newlines, spaces, and empty strings.",
            "partially_correct": "It splits text recursively starting with double newlines (\"\\n\\n\") and single newlines (\"\\n\").",
            "incorrect": "The default separator list is [\"###\", \"##\", \"#\"] to split Markdown headings first.",
            "correct_with_fluff": "Before chunking text in a RAG pipeline, you must install langchain-text-splitters via pip and configure embeddings and a vector database. RecursiveCharacterTextSplitter uses the default separator list [\"\\n\\n\", \"\\n\", \" \", \"\"] in order of double newlines, single newlines, spaces, and empty strings."
        }
    },
    {
        "test_id": "TC03_Splitter_Methods",
        "source_question_id": "Q114",
        "question": "According to the RecursiveCharacterTextSplitter documentation, which specific method should be called to obtain raw string chunks directly, and which method must be called to create LangChain Document objects?",
        "ground_truth": "To obtain string content directly, call `.split_text`. To create LangChain Document objects, call `.create_documents`.",
        "answers": {
            "correct": "Call `.split_text` to obtain raw string chunks directly, and `.create_documents` to create LangChain Document objects.",
            "partially_correct": "You can call `.split_text` to get plain strings back from the splitter.",
            "incorrect": "Call `.to_strings()` for raw strings and `.build_nodes()` for Document objects.",
            "correct_with_fluff": "Document loaders such as PyPDFLoader and WebBaseLoader read external data sources into memory before splitting. When using RecursiveCharacterTextSplitter, call `.split_text` for raw strings and `.create_documents` for Document objects."
        }
    }
]


def main():
    validation_results = []

    for tc in CONTROLLED_TEST_CASES:
        print(f"Running rubric validation for {tc['test_id']}...")
        ans = tc["answers"]

        # Comparison 1: Correct (A) vs Partially Correct (B)
        comp_1 = evaluate_pair(tc["question"], tc["ground_truth"], ans["correct"], ans["partially_correct"])

        # Comparison 2: Incorrect (A) vs Correct-with-Fluff (B)
        comp_2 = evaluate_pair(tc["question"], tc["ground_truth"], ans["incorrect"], ans["correct_with_fluff"])

        scores = {
            "correct": comp_1["answer_a"],
            "partially_correct": comp_1["answer_b"],
            "incorrect": comp_2["answer_a"],
            "correct_with_fluff": comp_2["answer_b"]
        }

        checks = {
            "correct_gt_partial_completeness": scores["correct"]["completeness"] > scores["partially_correct"]["completeness"],
            "partial_gt_incorrect_correctness": scores["partially_correct"]["factual_correctness"] > scores["incorrect"]["factual_correctness"],
            "correct_gt_fluff_relevance": scores["correct"]["relevance"] > scores["correct_with_fluff"]["relevance"],
            "fluff_preserves_correctness": scores["correct_with_fluff"]["factual_correctness"] >= 4
        }

        validation_results.append({
            "test_id": tc["test_id"],
            "source_question_id": tc["source_question_id"],
            "question": tc["question"],
            "scores": scores,
            "reasonings": {
                "correct_vs_partial": comp_1["reasoning"],
                "incorrect_vs_fluff": comp_2["reasoning"]
            },
            "validation_checks": checks,
            "passed_all_checks": all(checks.values())
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=4, ensure_ascii=False)

    print(f"\nRubric validation complete! Saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()