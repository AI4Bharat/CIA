import os
import re
import json
import argparse
import pandas as pd

OUTPUT_PATH = "/projects/data/llmteam/CIA/artifacts/results"


def stop_after_first_result(text):
    # Regular expression to match "[RESULT] x" where x is any number
    pattern = r"\[RESULT\] \d+"
    # Search for the pattern in the text
    match = re.search(pattern, text)
    try:
        if match:
            # Get the position of the end of the first match
            end_position = match.end()
            # Return the text up to the end of the first match
            return int(text[:end_position][-1])
        else:
            # If no match is found, return the original text
            return int(text[-1])
    except:
        return -1


def accuracy(predictions):
    ground_truth = list(predictions['ground_truth'].apply(stop_after_first_result))
    generated_label = list(predictions['generated_text'].apply(stop_after_first_result))

    if len(ground_truth) != len(generated_label):
        raise ValueError("The lengths of generated_label and ground_truth must be the same.")
    
    print(generated_label)
    print(ground_truth)
    correct_predictions = sum(1 for gen, true in zip(generated_label, ground_truth) if gen == true)
    total_predictions = len(ground_truth)
    
    return correct_predictions / total_predictions


def calculate_approx_accuracy(predictions):
    ground_truth = list(predictions['ground_truth'].apply(stop_after_first_result))
    generated_label = list(predictions['generated_text'].apply(stop_after_first_result))

    if len(ground_truth) != len(generated_label):
        raise ValueError("The lengths of generated_label and ground_truth must be the same.")
    
    correct_predictions = sum(1 for gen, true in zip(generated_label, ground_truth) if (gen == true) or (gen == true - 1 or gen == true + 1))
    total_predictions = len(ground_truth)
    
    accuracy = correct_predictions / total_predictions
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference using VLLM on multiple GPUs and save results to CSV")
    parser.add_argument("--model_name", type=str, required=True, help="name of the model")
    parser.add_argument("--lang", type=str, required=True, help="language to evaluate")
    args = parser.parse_args()
    
    predictions = pd.read_csv(f"{os.path.join(OUTPUT_PATH, args.model_name)}.tsv", sep='\t')
    acc_ = accuracy(predictions)
    approx_acc = calculate_approx_accuracy(predictions)

    with open(f"{OUTPUT_PATH}/{args.model_name}-result.json", 'w') as f:
        results = {
            'model': args.model_name,
            'accuracy': acc_,
            'approx accuracy': approx_acc
        }
        print(results)
        json.dump(results, f, indent=4)