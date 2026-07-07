import re
import pandas as pd

INPUT_FILE = "results/generated_answers.csv"
OUTPUT_FILE = "results/scored_generated_answers.csv"


def normalize_number_text(text):
    return str(text).replace(",", "").replace("$", "").lower()


def extract_numbers(text):
    text = normalize_number_text(text)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    return [float(m) for m in matches]


def numeric_match(ground_truth, llm_answer, tolerance=0.01):
    gt_numbers = extract_numbers(ground_truth)
    pred_numbers = extract_numbers(llm_answer)

    if not gt_numbers or not pred_numbers:
        return False

    gt = gt_numbers[0]

    for pred in pred_numbers:
        if gt == 0:
            if abs(pred - gt) <= tolerance:
                return True
        else:
            relative_error = abs(pred - gt) / abs(gt)
            if relative_error <= tolerance:
                return True

    return False


def groundedness_label(answer):
    answer_lower = str(answer).lower()

    if "missing information: none" in answer_lower:
        if "source used" in answer_lower:
            return "source_based"
        return "needs_manual_review"

    if "missing information" in answer_lower:
        return "grounded_refusal"

    if "source used" in answer_lower:
        return "source_based"

    return "needs_manual_review"

df = pd.read_csv(INPUT_FILE)

df["correct"] = df.apply(
    lambda row: numeric_match(row["ground_truth"], row["llm_answer"]),
    axis=1
)

df["groundedness_label"] = df["llm_answer"].apply(groundedness_label)

accuracy = df["correct"].mean()

print("Generated Answer Metrics")
print("------------------------")
print(f"Total answers scored: {len(df)}")
print(f"Factual Accuracy: {accuracy:.2%}")

print("\nGroundedness labels:")
print(df["groundedness_label"].value_counts())

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")
print(df[["uid", "ground_truth", "llm_answer", "correct", "groundedness_label"]])