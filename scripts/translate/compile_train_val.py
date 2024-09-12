import json
import pandas as pd
from tqdm import tqdm 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Batch processing')
    # data
    parser.add_argument('--split', help="train or test")
    parser.add_argument('--lang', help="language to translate to")
    args = parser.parse_args()
    return args

args = parse_args()
if args.split == 'train':
    _SPLIT = "Feedback-Collection"
elif args.split == 'test':
    _SPLIT = "Feedback-Bench"


original_testset_path = f"artifacts/{_SPLIT}/new_feedback_collection.json"
translated_instructions_path = f"/Users/sumanth/code/CIA/artifacts/batch/outputs/{_SPLIT}-instructions-en2{args.lang}-gpt-4o.jsonl"
translated_responses_path = f"/Users/sumanth/code/CIA/artifacts/batch/outputs/{_SPLIT}-responses-en2{args.lang}-gpt-4o.jsonl"

unq_instructions_path = f"artifacts/{_SPLIT}/unique_instructions.tsv"
unq_responses_path = f"artifacts/{_SPLIT}/unique_responses.tsv"

with open(translated_instructions_path) as f:
    translated_instructions = [json.loads(line) for line in f]
translated_instructions_df = pd.DataFrame(translated_instructions)

with open(translated_responses_path) as f:
    translated_responses = [json.loads(line) for line in f]
translated_responses_df = pd.DataFrame(translated_responses)

unq_instructions_df = pd.read_csv(unq_instructions_path, sep="\t")
unq_responses_df = pd.read_csv(unq_responses_path, sep="\t")

if args.split == 'train':
    with open(original_testset_path) as f:
        data = json.load(f)
elif args.split == 'test':
    with open(original_testset_path) as f:
        data = [json.loads(line) for line in f]


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



translated_feedback_collection = []
for d in tqdm(data):
    t_dict = {}
    try:
        resp_idx = unq_responses_df[unq_responses_df['response'] == d['orig_response']]['idx'].values[0]
        t_dict['orig_response'] = translated_responses_df[translated_responses_df['custom_id'] == resp_idx]['response'].values[0]['body']['choices'][0]['message']['content']

        inst_idx = unq_instructions_df[unq_instructions_df['instruction'] == d['orig_instruction']]['idx'].values[0]
        t_dict['orig_instruction'] = translated_instructions_df[translated_instructions_df['custom_id'] == inst_idx]['response'].values[0]['body']['choices'][0]['message']['content']

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

        _RUBRIC = (
            f"{t_dict['orig_criteria']}\n"
            f"Score 1: {t_dict['orig_score1_description']}\n"
            f"Score 2: {t_dict['orig_score2_description']}\n"
            f"Score 3: {t_dict['orig_score3_description']}\n"
            f"Score 4: {t_dict['orig_score4_description']}\n"
            f"Score 5: {t_dict['orig_score5_description']}\n"
        )
        t_dict['messages'] = [
            {"role": "user", "content": get_messages(t_dict['orig_instruction'], t_dict['orig_response'], t_dict['orig_reference_answer'], _RUBRIC)},
            {"role": "assistant", "content": t_dict['output'], }
        ]

        translated_feedback_collection.append(t_dict)
    except Exception as e:
        print(f"Error in {e}")


if args.split == "train":
    with open(f"artifacts/{_SPLIT}/{args.lang}_translated_feedback_collection.json", "w") as f:
        json.dump(translated_feedback_collection, f, indent=4, ensure_ascii=False)
elif args.split == "test":
     with open(f"artifacts/{_SPLIT}/{args.lang}_translated_feedback_bench.json", "w") as f:
        json.dump(translated_feedback_collection, f, indent=4, ensure_ascii=False)