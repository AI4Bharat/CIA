import json
import pandas as pd
import argparse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

base_parser = JsonOutputParser()

class CriteriaParser(BaseModel):
    criteria: str = Field(description="Criteria being considered")
    score1_description: str = Field(description="Description of score 1")
    score2_description: str = Field(description="Description of score 2")
    score3_description: str = Field(description="Description of score 3")
    score4_description: str = Field(description="Description of score 4")
    score5_description: str = Field(description="Description of score 5")
    
class ReferenceParser(BaseModel):
    answer: str = Field(description="Reference answer")
    
class ScoredResponseParser(BaseModel):
    response: str = Field(description="Response for the given score")
    feedback: str = Field(description="Feedback for the given response")


def criteria_parser(criteria_json: str) -> dict:
    """Parse the criteria json to get the question and criteria.
    
    
    Args:
        criteria_json (str): Criteria json
    
    Returns:
        dict: Question and criteria
    """
    # criteria = base_parser.invoke(criteria_json)
    criteria = JsonOutputParser(pydantic_object=CriteriaParser).invoke(criteria_json)
    return criteria


def reference_parser(reference_json: str) -> dict:
    """Parse the reference json to get the reference answer.
    
    
    Args:
        reference_json (str): Reference json
    
    Returns:
        dict: Reference answer
    """
    reference = JsonOutputParser(pydantic_object=ReferenceParser).invoke(reference_json)
    return reference

def scored_response_parser(scored_response_json: str) -> dict:
    """Parse the scored response json to get the response and feedback.
    
    
    Args:
        scored_response_json (str): Scored response json
    
    Returns:
        dict: Response and feedback
    """
    scored_response = JsonOutputParser(pydantic_object=ScoredResponseParser).invoke(scored_response_json)
    return scored_response