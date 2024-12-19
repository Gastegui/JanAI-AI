"""
Module for storing necessary models for LLM communication
"""
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
