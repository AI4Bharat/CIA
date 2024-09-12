import os
import json
import random
import argparse
import pandas as pd


_LANG_MAP = {
    'bn': 'Bengali',
    'hi': 'Hindi',
    'te': 'Telugu',
    'de': 'German',
    'fr': 'French',
    'ja': 'Japanese'
}


def create_jsonl(cdx: str, model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float, frequency_penalty: float, presence_penalty: float) -> dict:
    return {
        'custom_id': cdx,
        'method': 'POST',
        'url': '/v1/chat/completions',
        'body': {
            'model': f'{model_name}',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user',
                    'content': f'{prompt}'
                }
            ],
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'frequency_penalty': frequency_penalty,
            'presence_penalty': presence_penalty,
        }
    }


def dump_jsonl(args: argparse.Namespace, jsons: list, file_name: str) -> None:
    if args.debug:
        jsons = random.sample(jsons, 20)
    
    with open(file_name, 'w') as f:
        for json_ in jsons:
            f.write(json.dumps(json_) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description='Batch processing')
    # model arguments
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1, help='Top p for sampling')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens for sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0, help='Frequency penalty for sampling')
    parser.add_argument('--presence_penalty', type=float, default=0, help='Presence penalty for sampling')
    # debug
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    # mode
    parser.add_argument('--create_batch', action='store_true', help='Creates the jsonl file')
    parser.add_argument('--run_batch', action='store_true', help='Sends the batch to the server')
    # data
    parser.add_argument('--split', help="train or test")
    parser.add_argument('--lang', help="language to translate to")
    args = parser.parse_args()
    return args


def main(args):
    if args.create_batch:
        lang = args.lang
        if args.split == 'train':
            _SPLIT = "Feedback-Collection"
            with open(f"artifacts/{_SPLIT}/new_feedback_collection.json") as f:
                data = json.load(f)
        elif args.split == 'test':
            _SPLIT = "Feedback-Bench"
            with open(f"artifacts/{_SPLIT}/new_feedback_collection.json") as f:
                data = [json.loads(line) for line in f]

        instructions, responses = [], []
        for d in data:
            instructions.append(d['orig_instruction'])
            responses.append(d['orig_response'])

        # translate instructions
        instruction_jsons = []
        unq_inst_ids, unq_instructions = [], []
        for i, inst in enumerate(list(set(instructions))):
            PROMPT = (
                f"Translate the following Input Prompt from English to {_LANG_MAP[lang]}.\n"
                "Do not write any additional content, just translate the prompt.\n"
                "Input Prompt:\n"
                f"{inst}\n\n"
                f"{_LANG_MAP[lang]} Translation:\n"
            )
            dict_ = create_jsonl(f"inst-{i}", args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
            unq_inst_ids.append(f"inst-{i}")
            unq_instructions.append(inst)
            instruction_jsons.append(dict_)
        
        unq_instructions_df = pd.DataFrame({'idx': unq_inst_ids, 'instruction': unq_instructions})
        unq_instructions_df.to_csv(f"artifacts/{_SPLIT}/unique_instructions.tsv", sep="\t", index=False)
        dump_jsonl(args, instruction_jsons, f"artifacts/batch/inputs/{_SPLIT}-instructions-en2{lang}-{args.model}.jsonl")

        # translate responses
        response_jsons = []
        unq_resp_ids, unq_responses = [], []
        for i, resp in enumerate(list(set(responses))):
            PROMPT = (
                f"Translate the following Input Prompt from English to {_LANG_MAP[lang]}.\n"
                "Do not write any additional content, just translate the prompt.\n"
                "Input Prompt:\n"
                f"{resp}\n\n"
                f"{_LANG_MAP[lang]} Translation:\n"
            )
            dict_ = create_jsonl(f"resp-{i}", args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
            unq_resp_ids.append(f"resp-{i}")
            response_jsons.append(dict_)
            unq_responses.append(resp)

        unq_responses_df = pd.DataFrame({'idx': unq_resp_ids, 'response': unq_responses})
        unq_responses_df.to_csv(f"artifacts/{_SPLIT}/unique_responses.tsv", sep="\t", index=False)
        dump_jsonl(args, response_jsons, f"artifacts/batch/inputs/{_SPLIT}-responses-en2{lang}-{args.model}.jsonl")

if __name__ == '__main__':
    args = parse_args()
    main(args)

