"""
Module for storing data formats used while
communicating with Node-RED
"""

from pydantic import BaseModel


class UserBiometricInfo(BaseModel):
    """User biometric data shape for inputting in model"""

    height: float
    weight: float
    age: int
    activity_level: int
    gender: str


class Dish(BaseModel):
    """Data shape for dishes to be recommended"""

    name: str
    ingredient_components: list[str]
    calories: float
    carbohydrates: float
    protein: float
    fibers: float
    fats: float
    saturated_fats: float
    sugars: float
    added_sugars: float
    LDL_cholesterol: float
    VDL_cholesterol: float


class IntakePredictionIn(BaseModel):
    """Input data expectation for intake model input"""

    info: UserBiometricInfo


class ImageScanIn(BaseModel):
    """Input data expectation for food recognition model input"""

    input: str


class RecommendationIn(BaseModel):
    """Input data expectation for recommendation model input"""

    input: str
