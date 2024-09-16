import argparse
import json
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser

from utils import create_jsonl, dump_jsonl
from parsers import ReferenceParser

 


def get_reference_answer(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a question, and a criteria, get the appropriate reference answer for that question.
    
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=ReferenceParser)
        PROMPT = (
            f"Your job is to generate a response for the given instruction that would get a score of 5 on the given score rubric.\n\n"
            "Instruction:\n"
            f"{row['input']}\n\n"
            "Scoring Rubric:\n"
            f"{row['specific_rubric']}\n\n"
            "- The response should be a response that would get a score of 5 from the score rubric.\n"
            "- The response should be as detailed as possible unless the score rubric is related to conciseness or brevity. It should consist of multiple paragraphs, a list of items, or a step-by-step reasoning process.\n"
            "- The response should look like how a well-prompted GPT-4 would normally answer your problem.\n"
            "- Do not explicitly state the keywords of the score rubric inside the response.\n\n"
            "Please provide the response in the json format as mentioned below.\n"
            f"{parser.get_format_instructions()}\n\n"
        )
        dict_ = create_jsonl(row['id'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/reference-answers-{args.temperature}.jsonl')
    return
    

def parse_args():
    parser = argparse.ArgumentParser(description="Get rubrics for scoring responses")
    parser.add_argument("--testset_path", type=str, help="Path to testset")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    # model arguments
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1, help='Top p for sampling')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens for sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0, help='Frequency penalty for sampling')
    parser.add_argument('--presence_penalty', type=float, default=0, help='Presence penalty for sampling')
    
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    testset = pd.read_csv(args.testset_path, sep='\t')
    
    get_reference_answer(args, testset)
        
    
    

        