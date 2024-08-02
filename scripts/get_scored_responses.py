import argparse
import json
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser

from utils import create_jsonl, dump_jsonl
from parsers import ScoredResponseParser

 


def get_scored_answer(args: argparse.Namespace, testset: pd.DataFrame, score: int) -> None:
    """Given a question, and a criteria, get the appropriate reference answer for that question.
    
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=ScoredResponseParser)
        PROMPT = (
            f"Your job is to generate a response that would get a score of {score} and corresponding feedback based on the given score rubric. "
            "For reference, a reference response that would get a score of 5 is also given.\n\n"
            "Instruction:\n"
            f"{row['input']}\n\n"
            "Scoring Rubric:\n"
            f"{row['specific_rubric']}\n\n"
            "Reference response (Score 5):\n"
            f"{row['reference_answer']}\n\n"
            "* Response\n"
            f"- The quality of the score {score} response should be determined based on the score rubric, not by its length.\n"
            f"- The score {score} response should have the same length as the reference response.\n"
            "- Do not explicitly state the keywords of the score rubric inside the response.\n\n"
            "* Feedback\n"
            f"- The score {score} feedback should each be an explanation of why the response would get a score of {score}. It should be written based on the generated response and score rubric.\n"
            f"- The score {score} feedback shouldn’t just copy and paste the score rubric, but it should also give very detailed feedback on the content of the corresponding response.\n"
            f"- The score {score} feedback should include the phrase ”So the overall score is {score}” in the last sentence.\n"
            "Please provide the response in the json format as mentioned below.\n"
            f"{parser.get_format_instructions()}\n\n"
        )
        dict_ = create_jsonl(row['id'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/{score}-answers-{args.temperature}.jsonl')
    return

    

def parse_args():
    parser = argparse.ArgumentParser(description="Get rubrics for scoring responses")
    parser.add_argument("--testset_path", type=str, help="Path to testset")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    # model arguments
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1, help='Top p for sampling')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens for sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0, help='Frequency penalty for sampling')
    parser.add_argument('--presence_penalty', type=float, default=0, help='Presence penalty for sampling')
    
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    testset = pd.read_csv(args.testset_path, sep='\t')
    
    for i in range(1, 6):
        get_scored_answer(args, testset, i)
        
    
    

        