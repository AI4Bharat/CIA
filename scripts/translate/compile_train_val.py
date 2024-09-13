import json
import pandas as pd
import argparse
import glob
from tqdm import tqdm 
from joblib import Parallel, delayed


def get_messages(instruction, response, reference_answer, rubric):
    ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."

    ABSOLUTE_PROMPT = f"""###Task Description:
    An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
    1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
    3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
    4. Please do not generate any other opening, closing, and explanations.

    ###The instruction to evaluate:
    {instruction}

    ###Response to evaluate:
    {response}

    ###Reference Answer (Score 5):
    {reference_answer}

    ###Score Rubrics:
    {rubric}

    ###Feedback: """
    return ABS_SYSTEM_PROMPT + "\n\n" + ABSOLUTE_PROMPT

def load_translated_data(translated_path):
    files = sorted(glob.glob(translated_path))
    translated_data = []
    for file in files:
        with open(file) as f:
            for line in f:
                translated_data.append(json.loads(line))
    return pd.DataFrame(translated_data)

def process_entry(d, unq_instructions_df, unq_responses_df, translated_instructions_df, translated_responses_df):
    t_dict = {}
    try:
        # Find response index
        resp_idx = unq_responses_df[unq_responses_df['response'] == d['orig_response']]['idx'].values[0]
        t_dict['orig_response'] = translated_responses_df[translated_responses_df['custom_id'] == resp_idx]['response'].values[0]['body']['choices'][0]['message']['content']

        # Find instruction index
        inst_idx = unq_instructions_df[unq_instructions_df['instruction'] == d['orig_instruction']]['idx'].values[0]
        t_dict['orig_instruction'] = translated_instructions_df[translated_instructions_df['custom_id'] == inst_idx]['response'].values[0]['body']['choices'][0]['message']['content']

        # Collect additional fields
        t_dict['orig_criteria'] = d['orig_criteria']
        t_dict['orig_score1_description'] = d['orig_score1_description']
        t_dict['orig_score2_description'] = d['orig_score2_description']
        t_dict['orig_score3_description'] = d['orig_score3_description']
        t_dict['orig_score4_description'] = d['orig_score4_description']
        t_dict['orig_score5_description'] = d['orig_score5_description']
        t_dict['orig_reference_answer'] = d['orig_reference_answer']
        t_dict['orig_feedback'] = d['orig_feedback']
        t_dict['orig_score'] = d['orig_score']
        t_dict['input'] = d['input']
        t_dict['output'] = d['output']

        # Create rubric string
        _RUBRIC = (
            f"{t_dict['orig_criteria']}\n"
            f"Score 1: {t_dict['orig_score1_description']}\n"
            f"Score 2: {t_dict['orig_score2_description']}\n"
            f"Score 3: {t_dict['orig_score3_description']}\n"
            f"Score 4: {t_dict['orig_score4_description']}\n"
            f"Score 5: {t_dict['orig_score5_description']}\n"
        )
        
        # Prepare messages
        t_dict['messages'] = [
            {"role": "user", "content": get_messages(t_dict['orig_instruction'], t_dict['orig_response'], t_dict['orig_reference_answer'], _RUBRIC)},
            {"role": "assistant", "content": t_dict['output']}
        ]
        
        return t_dict
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_data(data, unq_instructions_df, unq_responses_df, translated_instructions_df, translated_responses_df):
    translated_feedback_collection = Parallel(n_jobs=1)(
        delayed(process_entry)(d, unq_instructions_df, unq_responses_df, translated_instructions_df, translated_responses_df)
        for d in tqdm(data)
    )
    print("Number of entries:", len(translated_feedback_collection))
    translated_feedback_collection = [t_dict for t_dict in translated_feedback_collection if t_dict is not None]
    print("Number of entries after filtering:", len(translated_feedback_collection))
    
    return translated_feedback_collection

def parse_args():
    parser = argparse.ArgumentParser(description='Batch processing')
    # data
    parser.add_argument('--split', help="train or test")
    parser.add_argument('--lang', help="language to translate to")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.split == 'train':
        _SPLIT = "Feedback-Collection"
    elif args.split == 'test':
        _SPLIT = "Feedback-Bench"
        
    original_testset_path = f"artifacts/{_SPLIT}/new_feedback_collection.json"
    translated_instructions_path = f"artifacts/batch/outputs/{args.lang}/{_SPLIT}-instructions-en2{args.lang}-gpt-4o-2024-08-06*.jsonl"
    translated_responses_path = f"artifacts/batch/outputs/{args.lang}/{_SPLIT}-responses-en2{args.lang}-gpt-4o-2024-08-06*.jsonl"

    unq_instructions_path = f"artifacts/{_SPLIT}/unique_instructions.tsv"
    unq_responses_path = f"artifacts/{_SPLIT}/unique_responses.tsv"
    
    print("Loading data...")
    
    translated_instructions_df = load_translated_data(translated_instructions_path)
    translated_responses_df = load_translated_data(translated_responses_path)
    
    unq_instructions_df = pd.read_csv(unq_instructions_path, sep="\t")
    unq_responses_df = pd.read_csv(unq_responses_path, sep="\t")
    
    if args.split == 'train':
        with open(original_testset_path) as f:
            data = json.load(f)
    elif args.split == 'test':
        with open(original_testset_path) as f:
            data = [json.loads(line) for line in f]

    translated_feedback_collection = process_data(
        data,
        unq_instructions_df,
        unq_responses_df,
        translated_instructions_df,
        translated_responses_df
        )
    
    if args.split == "train":
        with open(f"artifacts/final_upload/xx-prometheus/data/{args.lang}_translated_feedback_collection.json", "w") as f:
            json.dump(translated_feedback_collection, f, indent=4, ensure_ascii=False)
            
    elif args.split == "test":
        with open(f"artifacts/final_upload/xx-prometheus/data/{args.lang}_translated_feedback_bench.json", "w") as f:
            json.dump(translated_feedback_collection, f, indent=4, ensure_ascii=False)
            
if __name__ == '__main__':
    main()