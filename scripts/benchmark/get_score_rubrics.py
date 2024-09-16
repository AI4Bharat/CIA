import argparse
import json
import pandas as pd

from langchain_core.output_parsers import JsonOutputParser

from utils import create_jsonl, dump_jsonl
from parsers import CriteriaParser



    

    
specific_question_criteria_list = [
    {
        "question": "Is using a VPN really necessary for everyone who uses the internet, given that it encrypts the internet traffic and supposedly protects your online activities from being monitored or tracked by third parties?",
        "criteria": {
            'criteria': 'Does the response evaluate the necessity of VPN usage for all internet users, considering both privacy advantages and potential limitations?', 
            'score1_description': 'The response incorrectly suggests that VPNs are either essential for all internet users without addressing privacy expectations or assumes VPNs offer no real benefits, lacking an understanding of how VPNs work.', 
            'score2_description': 'The response recognizes the privacy benefits of using a VPN but overstates its effectiveness or fails to mention its limitations and potential issues.', 
            'score3_description': 'The response mentions the benefits of using a VPN for privacy but provides an incomplete view of why it might not be necessary for everyone, ignoring diverse user needs and potential drawbacks.', 
            'score4_description': 'The response accurately discusses the privacy benefits of using a VPN and also acknowledges its limitations, but it doesn’t sufficiently detail the circumstances under which using a VPN might be more or less necessary.', 
            'score5_description': 'The response provides an excellent analysis by highlighting the significance of VPNs in enhancing privacy for certain users while noting their limitations and the importance of understanding individual needs and risks. It suggests evaluating privacy concerns against the potential impact on internet speed and access to localized content.'
            }
    },
    {
        "question": "Elena, a local community leader, is tasked with revitalizing a once-popular neighborhood park that has fallen into disuse. The park's decline has been attributed to outdated facilities, lack of engaging activities, and safety concerns after dark. Elena believes that with the right improvements, the park can become a vibrant community space once again. She aims to attract a diverse group of visitors, including families, fitness enthusiasts, and nature lovers, by addressing these issues and adding new features that the community values.  Question: What strategic plan should Elena develop to rejuvenate the neighborhood park and ensure it meets the community's needs and safety standards?",
        "criteria": {
            'criteria': "Does the response correctly provide a comprehensive strategic plan that addresses facilities' improvements, engaging activities, and safety measures?", 
            'score1_description': 'The response fails to offer a coherent strategy, neglecting either facility improvements, activity engagement, or safety measures.', 
            'score2_description': 'The response provides a basic framework for addressing one or two aspects of the revitalization (facilities, activities, or safety), but lacks an integrated approach.', 
            'score3_description': 'The response outlines a plan that addresses facility improvements, activity engagement, and safety measures, but with limited depth or creativity.', 
            'score4_description': 'The response proposes a well-rounded strategy for park revitalization, covering improvements to facilities, a variety of activities, and safety measures, though it may overlook some community-specific needs.', 
            'score5_description': "The response delivers a detailed and creative strategic plan that fully addresses facilities' improvements, diverse engaging activities, and comprehensive safety measures, aligning with community values and needs. The plan includes methods for community involvement and feedback."
            }
    },
    {
        "question": "I teach a mixed martial arts class and one of my students is an observant Sikh. Are there any specific considerations or accommodations I should be aware of to ensure that I respect his cultural practices and beliefs in the class?",
        "criteria": {
            'criteria': 'Does the response sensitively and accurately address cultural and religious considerations without resorting to stereotypes or generalizations?', 
            'score1_description': 'The response includes harmful stereotypes or misrepresentations about Sikh practices and beliefs, showing a significant lack of cultural sensitivity and understanding.', 
            'score2_description': 'The response makes broad generalizations about Sikh practices without acknowledging individual variation or the need for direct communication, leading to potential cultural insensitivity.', 
            'score3_description': 'The response mentions specific Sikh practices but lacks depth or nuance, risking oversimplification or misunderstanding of these practices.', 
            'score4_description': 'The response addresses Sikh practices with respect but could benefit from emphasizing the importance of direct communication with the individual for personal accommodations.', 
            'score5_description': 'The response thoughtfully and accurately considers Sikh practices and beliefs, emphasizing respect, individual accommodations, and the importance of direct communication, without resorting to stereotypes or generalizations.'
            }
    }
]

general_question_criteria_list = [
    {
        "question": "Is using a VPN really necessary for everyone who uses the internet, given that it encrypts the internet traffic and supposedly protects your online activities from being monitored or tracked by third parties?",
        "criteria": {
            "criteria": "Does the response effectively address the question, providing a balanced view of the topic while considering both benefits and potential limitations?",
            "score1_description": "The response fails to address key aspects of the question, offering an overly simplistic or one-sided view without sufficient explanation or consideration of different perspectives.",
            "score2_description": "The response acknowledges some relevant points but lacks depth, either overstating certain aspects or not fully exploring the implications of the topic.",
            "score3_description": "The response provides a general understanding of the topic, mentioning key benefits and drawbacks but with some gaps in coverage or minor inaccuracies.",
            "score4_description": "The response offers a well-rounded analysis, clearly outlining the pros and cons of the topic and addressing most relevant factors, though it may lack in specific details or additional insights.",
            "score5_description": "The response delivers an excellent and thorough analysis, offering a comprehensive discussion that balances the advantages and limitations, while providing insightful context and addressing different perspectives effectively."
        }
    },
    {
        "question": "Elena, a local community leader, is tasked with revitalizing a once-popular neighborhood park that has fallen into disuse. The park's decline has been attributed to outdated facilities, lack of engaging activities, and safety concerns after dark. Elena believes that with the right improvements, the park can become a vibrant community space once again. She aims to attract a diverse group of visitors, including families, fitness enthusiasts, and nature lovers, by addressing these issues and adding new features that the community values.  Question: What strategic plan should Elena develop to rejuvenate the neighborhood park and ensure it meets the community's needs and safety standards?",
        "criteria": {
            "criteria": "Does the response provide a comprehensive and actionable strategic plan that addresses the key aspects of the problem and aligns with the needs?",
            "score1_description": "The response lacks a coherent strategy, failing to address key aspects of the problem or providing only vague and impractical suggestions.",
            "score2_description": "The response identifies some key aspects of the problem and offers basic strategies, but it is incomplete or lacks integration between the different elements of the plan.",
            "score3_description": "The response presents a strategic plan that covers the essential aspects of the problem with some actionable steps, but may lack detail, creativity, or a clear connection to general needs.",
            "score4_description": "The response provides a well-rounded and thoughtful strategic plan that addresses the main issues with clear, actionable steps. It is generally aligned with general needs, but might miss some innovative or specific opportunities for improvement.",
            "score5_description": "The response offers an in-depth, creative, and well-integrated strategic plan that fully addresses all key aspects of the problem. The plan is practical, aligns closely with general needs, and includes mechanisms for ensuring general involvement and adaptability."
        }
    },
    {
        "question": "I teach a mixed martial arts class and one of my students is an observant Sikh. Are there any specific considerations or accommodations I should be aware of to ensure that I respect his cultural practices and beliefs in the class?",
        "criteria": {
            "criteria": "Does the response appropriately address the cultural or religious considerations with sensitivity, accuracy, and respect?",
            "score1_description": "The response demonstrates a lack of cultural awareness, possibly including stereotypes or misrepresentations, and fails to show respect for individual beliefs.",
            "score2_description": "The response acknowledges some cultural or religious practices but relies on generalizations, without fully understanding or respecting the nuances and individual variations in these practices.",
            "score3_description": "The response provides an overview of relevant cultural or religious practices with some accuracy but may lack depth or a nuanced understanding, potentially leading to oversimplification.",
            "score4_description": "The response shows a good understanding of cultural or religious considerations and addresses them with respect. It could, however, benefit from a stronger emphasis on the importance of direct communication with the individual to ensure personalized accommodations.",
            "score5_description": "The response is culturally sensitive and accurate, offering thoughtful and respectful consideration of the individual’s practices and beliefs. It emphasizes the importance of direct communication to tailor accommodations without relying on generalizations or stereotypes."
        }
    }
]


def get_specific_rubrics(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a question, get the appropriate question specific rubrics for evaluating that question.
    
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=CriteriaParser)
        PROMPT = (
            "We are brainstorming criteria with which to grade a language model on its responses in diverse situations. "
            "A 'criteria' is some useful, real-world objective, and associated rubric for scores 1-5, that tests a capability. "
            "Below are a few examples of questions and their associated criteria in json format.\n\n"
            "Question 1:\n"
            f"{json.dumps(specific_question_criteria_list[0]['question'])}\n"
            "Critera for Question 1:\n"
            f"```{json.dumps(specific_question_criteria_list[0]['criteria'])}```\n\n"
            "Question 2:\n"
            f"{json.dumps(specific_question_criteria_list[1]['question'])}\n"
            "Critera for Question 2:\n"
            f"```{json.dumps(specific_question_criteria_list[1]['criteria'])}```\n\n"
            "Question 3:\n"
            f"{json.dumps(specific_question_criteria_list[2]['question'])}\n"
            "Critera for Question 3:\n"
            f"```{json.dumps(specific_question_criteria_list[2]['criteria'])}```\n\n"
            "Below is a new question. Please brainstorm a new criteria and scoring rubrics for this question.\n\n"
            "Question: \n"
            f"{row['input']}\n\n"
            "Be creative and create new but useful criteria that people would practically evaluate. "
            "Please format the criteria in a json as mentioned below (same as the above examples with no extra or surrounding text). Give only the criteria and score descriptions and nothing else.\n"
            f"{parser.get_format_instructions()}\n\n"
        )
        dict_ = create_jsonl(row['id'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/specific-rubric-{args.temperature}.jsonl')
    return

def get_general_rubrics(args: argparse.Namespace, testset: pd.DataFrame) -> None:
    """Given a question, get the appropriate general rubrics for evaluating that question.
    
    
    Args:
        args (argparse.Namespace): Arguments
        testset (pd.DataFrame): Testset
        
    Returns:
        None
    """
    jsons = []
    for _, row in testset.iterrows():
        parser = JsonOutputParser(pydantic_object=CriteriaParser)
        PROMPT = (
            "We are brainstorming criteria with which to grade a language model on its responses in diverse situations. "
            "A 'criteria' is some useful, real-world objective, and associated rubric for scores 1-5, that tests a capability. "
            "Below are a few examples of questions and their associated criteria in json format.\n\n"
            "Question 1:\n"
            f"{json.dumps(general_question_criteria_list[0]['question'])}\n"
            "Critera for Question 1:\n"
            f"```{json.dumps(general_question_criteria_list[0]['criteria'])}```\n\n"
            "Question 2:\n"
            f"{json.dumps(general_question_criteria_list[1]['question'])}\n"
            "Critera for Question 2:\n"
            f"```{json.dumps(general_question_criteria_list[1]['criteria'])}```\n\n"
            "Question 3:\n"
            f"{json.dumps(general_question_criteria_list[2]['question'])}\n"
            "Critera for Question 3:\n"
            f"```{json.dumps(general_question_criteria_list[2]['criteria'])}```\n\n"
            "Below is a new question. Please brainstorm a new criteria and scoring rubrics for this question.\n\n"
            "Question: \n"
            f"{row['input']}\n\n"
            "Be creative and create new but useful criteria that people would practically evaluate. "
            "Dont give very question specific rubrics. Give general rubrics that can be used for evaluating responses of other questions of similar type (similar intent as the current question). "
            "These rubrics should ideally evaluate the broad intent covered in the given question without going into the exact question level specifics."
            "Please format the criteria in a json as mentioned below (same as the above examples with no extra or surrounding text). Give only the criteria and score descriptions and nothing else.\n"
            f"{parser.get_format_instructions()}\n\n"
        )
        dict_ = create_jsonl(row['id'], args.model, PROMPT, args.max_tokens, args.temperature, args.top_p, args.frequency_penalty, args.presence_penalty)
        jsons.append(dict_)
    
    dump_jsonl(args, jsons, f'{args.data_dir}/general-rubrics-{args.temperature}.jsonl')
    return
    

def parse_args():
    parser = argparse.ArgumentParser(description="Get rubrics for scoring responses")
    parser.add_argument("--type", type=str, help="Type of rubrics to get", choices=['general', 'specific', 'both'])
    parser.add_argument("--testset_path", type=str, help="Path to testset")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    # model arguments
    parser.add_argument('--model', type=str, default='gpt-4o', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=1, help='Top p for sampling')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Max tokens for sampling')
    parser.add_argument('--frequency_penalty', type=float, default=0, help='Frequency penalty for sampling')
    parser.add_argument('--presence_penalty', type=float, default=0, help='Presence penalty for sampling')
    
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    testset = pd.read_csv(args.testset_path, sep='\t')
    
    if args.debug:
        testset = testset.head(20)
    
    if args.type == 'general':
        get_general_rubrics(args, testset)
    elif args.type == 'specific':
        get_specific_rubrics(args, testset)
    elif args.type == 'both':
        get_general_rubrics(args, testset)
        get_specific_rubrics(args, testset)
        
    
    

        