import json
import sys
import time
from pathlib import Path

# Find the project root
ROOT = Path(__file__).resolve().parents[3]

# Import the existing LLM judge
sys.path.insert(0, str(ROOT / "02_RAG_Pipeline"))

from evaluation_script2 import judge_answers

# Input and output files
INPUT_FILE = ROOT / "02_RAG_Pipeline" / "evaluation_results1.json"
OUTPUT_FILE = Path(__file__).parent / "position_bias_results.json"

# Load existing evaluation results (read-only)
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    records = json.load(file)

# Start with five questions
selected_questions = records[:5]

position_bias_results = []

for record in selected_questions:
    question_id = record["id"]

    print(f"Testing question {question_id}...")

    question = record["question"]
    ground_truth = record["ground_truth"]
    base_answer = record["base"]["answer"]
    rag_answer = record["rag"]["answer"]

    # Run 1: Base answer appears first
    base_first = judge_answers(
        question,
        ground_truth,
        base_answer,
        rag_answer,
        presentation_order="base_first"
    )

    time.sleep(5)

    # Run 2: RAG answer appears first
    rag_first = judge_answers(
        question,
        ground_truth,
        base_answer,
        rag_answer,
        presentation_order="rag_first"
    )

    # Compare the scores
    differences = {}

    for model in ["base", "rag"]:
        differences[model] = {}

        for metric in [
            "factual_correctness",
            "completeness",
            "relevance"
        ]:
            differences[model][metric] = abs(
                base_first[model][metric]
                - rag_first[model][metric]
            )

    position_bias_results.append({
        "question_id": question_id,
        "base_first": base_first,
        "rag_first": rag_first,
        "score_differences": differences
    })

    # Save progress after each question
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(position_bias_results, file, indent=4)

    print(f"Completed question {question_id}")

    time.sleep(5)

print("Position bias testing completed!")
print(f"Results saved to: {OUTPUT_FILE}")