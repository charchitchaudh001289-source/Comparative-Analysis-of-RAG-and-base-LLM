import json
import statistics
import time
from pathlib import Path
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

# ------------------------------------------------------------
# CONFIGURATION (Matches evaluation_script2.py)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR.parent / "02_RAG_Pipeline" / "sg_queryR1.json"
OUTPUT_PATH = BASE_DIR / "consistency_results_for_sg.json"
JUDGE_MODEL = "gemini-3.1-flash-lite"
#TARGET_IDS = ["Q101", "Q104", "Q106", "Q108", "Q113"]  # Representative mix across categories
TARGET_IDS = None  # Set to None to run all questions, or ["Q101", "Q104", ...] for a subset

client = genai.Client()


def generate_judge_response(prompt: str):
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            # Stay under the 15 RPM Free Tier limit (60s / 15 = 4s per request)
            time.sleep(4.5)
            return response

        except ClientError as error:
            if getattr(error, "code", None) == 429 and attempt < max_retries - 1:
                print("Rate limit hit (429). Waiting 60 seconds before retrying...")
                time.sleep(60)
            else:
                raise

        except ServerError as error:
            if getattr(error, "code", None) == 503 and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"Gemini temporarily unavailable (503). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


def judge_fixed_order(question: str, ground_truth: str, base_answer: str, rag_answer: str) -> dict:
    """Evaluates with FIXED order: Answer A = Base, Answer B = RAG."""
    prompt = f"""
You are an impartial evaluator comparing two answers to a question.

Evaluate both answers using the supplied ground truth.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

ANSWER A:
{base_answer}

ANSWER B:
{rag_answer}

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
    result = json.loads(response.text)
    return {
        "base": result["answer_a"],
        "rag": result["answer_b"],
        "reasoning": result["reasoning"],
        "presentation_order": ["base", "rag"]
    }


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    #selected = [item for item in all_results if item["id"] in TARGET_IDS]
    selected = all_results if not TARGET_IDS else [item for item in all_results if item["id"] in TARGET_IDS]
    if not selected:
        selected = all_results[:5]

    consistency_output = []
    criteria = ["factual_correctness", "completeness", "relevance"]

    for item in selected:
        q_id = item["id"]
        print(f"Running 3x consistency check for {q_id} ({item['category']})...")

        runs = []
        for run_num in range(1, 4):
            eval_res = judge_fixed_order(
                question=item["question"],
                ground_truth=item["ground_truth"],
                base_answer=item["base"]["answer"],
                rag_answer=item["rag"]["answer"]
            )
            runs.append({
                "run": run_num,
                "base": eval_res["base"],
                "rag": eval_res["rag"],
                "reasoning": eval_res["reasoning"]
            })

        # Calculate consistency statistics (mean, std_dev, range) across the 3 runs
        stats_summary = {"base": {}, "rag": {}}
        for system in ["base", "rag"]:
            for crit in criteria:
                scores = [r[system][crit] for r in runs]
                stats_summary[system][crit] = {
                    "scores": scores,
                    "mean": round(statistics.mean(scores), 4),
                    "std_dev": round(statistics.stdev(scores), 4),
                    "range": max(scores) - min(scores)
                }

        consistency_output.append({
            "id": q_id,
            "category": item["category"],
            "question": item["question"],
            "presentation_order": ["base", "rag"],
            "runs": runs,
            "consistency_stats": stats_summary
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(consistency_output, f, indent=4, ensure_ascii=False)

    print(f"\nConsistency validation complete! Saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()