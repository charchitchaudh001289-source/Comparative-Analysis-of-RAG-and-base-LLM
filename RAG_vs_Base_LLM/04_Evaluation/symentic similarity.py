import json
import re
import random
import time

from pathlib import Path
from collections import Counter

from sentence_transformers import SentenceTransformer, util

from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError


INPUT_PATH = Path(
    r"/Users/smritigurung/Downloads/Gaming /Comparative-Analysis-of-RAG-and-base-LLM/RAG_vs_Base_LLM/02_RAG_Pipeline/sg_queryR1.json"
)

OUTPUT_PATH = Path(
    r"/Users/smritigurung/Downloads/Gaming /Comparative-Analysis-of-RAG-and-base-LLM/RAG_vs_Base_LLM/02_RAG_Pipeline/EvaResult.json"
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

JUDGE_MODEL = "gemini-3.1-flash-lite"


client = genai.Client()

print("Loading sentence embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Embedding model loaded successfully.")


def load_results():

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list):
        raise ValueError("The input JSON must contain a list.")

    print(f"Loaded {len(results)} questions.")

    return results


def normalize_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    text = " ".join(text.split())

    return text


def exact_match(prediction, ground_truth):

    prediction = normalize_text(prediction)
    ground_truth = normalize_text(ground_truth)

    if prediction == ground_truth:
        return 1

    return 0


def token_f1(prediction, ground_truth):

    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    prediction_counts = Counter(prediction_tokens)
    ground_truth_counts = Counter(ground_truth_tokens)

    common_words = prediction_counts & ground_truth_counts

    overlap = sum(common_words.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)

    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def semantic_similarity(prediction, ground_truth):

    prediction_embedding = embedding_model.encode(
        prediction,
        convert_to_tensor=True
    )

    ground_truth_embedding = embedding_model.encode(
        ground_truth,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        prediction_embedding,
        ground_truth_embedding
    )

    return float(similarity.item())


def generate_judge_response(prompt):

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

            return response

        except (ClientError, ServerError) as error:

            error_code = getattr(error, "code", None)

            # Retry only rate limits (429) and temporary server errors (503)

            if error_code not in (429, 503):
                raise

            if attempt == max_retries - 1:
                raise

            if error_code == 429:
                wait_time = 65
                print(
                    f"Gemini rate limit reached (429). "
                    f"Waiting {wait_time} seconds..."
                )

            else:
                wait_time = 2 ** (attempt + 1)
                print(
                    f"Gemini temporarily unavailable (503). "
                    f"Retrying in {wait_time} seconds..."
                )

            time.sleep(wait_time)


def judge_answers(
    question,
    ground_truth,
    base_answer,
    rag_answer,
    presentation_order="random"
):

    answers = [
        ("base", base_answer),
        ("rag", rag_answer)
    ]

    # Randomize answer order to reduce position bias in the judge
    if presentation_order == "random":
        random.shuffle(answers)

    elif presentation_order == "base_first":
        pass

    elif presentation_order == "rag_first":
        answers.reverse()

    else:
        raise ValueError(
            "presentation_order must be 'random', 'base_first', or 'rag_first'"
        )

    answer_a = answers[0][1]
    answer_b = answers[1][1]

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

    judge_result = json.loads(response.text)

    scores = {}

    scores[answers[0][0]] = judge_result["answer_a"]
    scores[answers[1][0]] = judge_result["answer_b"]

    return {
        "base": scores["base"],
        "rag": scores["rag"],
        "reasoning": judge_result["reasoning"],

        "presentation_order": [
            answers[0][0],
            answers[1][0]
        ]
    }


def evaluate_results(results):

    evaluated_results = []

    total_questions = len(results)

    for index, item in enumerate(results):

        print(
            f"\nEvaluating question {index + 1}/{total_questions}..."
        )

        question = item["question"]
        ground_truth = item["ground_truth"]

        base_answer = item["base"]["answer"]
        rag_answer = item["rag"]["answer"]


        base_exact = exact_match(base_answer, ground_truth)
        base_f1 = token_f1(base_answer, ground_truth)
        base_similarity = semantic_similarity(
            base_answer,
            ground_truth
        )

        rag_exact = exact_match(rag_answer, ground_truth)
        rag_f1 = token_f1(rag_answer, ground_truth)
        rag_similarity = semantic_similarity(
            rag_answer,
            ground_truth
        )


        judge_result = judge_answers(
            question,
            ground_truth,
            base_answer,
            rag_answer
        )


        evaluated_item = item.copy()

        evaluated_item["evaluation"] = {

            "base": {
                "exact_match": base_exact,
                "token_f1": base_f1,
                "semantic_similarity": base_similarity
            },

            "rag": {
                "exact_match": rag_exact,
                "token_f1": rag_f1,
                "semantic_similarity": rag_similarity
            },

            "llm_judge": judge_result
        }

        evaluated_results.append(evaluated_item)

        # Save after every question so completed work survives a crash
        save_results(evaluated_results)

        print("Question evaluation completed.")

    return evaluated_results


def save_results(results):

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )


def main():

    print("\nStarting evaluation...\n")

    results = load_results()

    evaluated_results = evaluate_results(results)

    save_results(evaluated_results)

    print("\n===================================")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("===================================")

    print(f"Total questions evaluated: {len(evaluated_results)}")

    print(f"Results saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
