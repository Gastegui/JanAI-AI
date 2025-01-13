"""
Module for storing necessary models for LLM communication
"""
from typing import List

from pydantic import BaseModel


class RequestLlm(BaseModel):
    """
    Model for validating proper LLM input.

    Attributes:
        userID (int): The user ID.
    """

    userID: int


class ResponseLlm(BaseModel):
    """
    Model for validating proper LLM output.

    Attributes:
        calorie_prediction (float): The predicted calorie intake.
    """

    calorie_prediction: float


class RequestDlm(BaseModel):
    """Input data expectation for food recognition model input.

    Attributes:
        input (str): The input string representing food data.
    """

    input: str


class ResponseDlm(BaseModel):
    """Input data expectation for recommendation model input.

    Attributes:
        predicted_class (str): The predicted class name.
        confidence (float): The confidence score of the prediction.
        all_predictions (List[tuple]): A list of tuples containing class names and their probabilities.
    """

    predicted_class: str
    confidence: float
    all_predictions: List[tuple]
