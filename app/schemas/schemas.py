"""
Module for storing necessary models for LLM communication
"""
from typing import List

from pydantic import BaseModel


class RequestLlm(BaseModel):
    """
    Model for validating proper LLM input
    """

    userID: int


class ResponseLlm(BaseModel):
    """
    Model for validating proper LLM output
    """

    calorie_prediction: float


class RequestDlm(BaseModel):
    """Input data expectation for food recognition model input"""

    input: str


class ResponseDlm(BaseModel):
    """Input data expectation for recommendation model input"""

    predicted_class: str
    confidence: float
    all_predictions: List[tuple]
