"""
Module file dedicated to storing model classes
"""
from ..dto import Dish


class ImageModel:
    """
    Model class to encapsulate image recognition logic
    """

    def predict(self, file):
        """
        Actual model is still a TO-DO --- TEMPORARY METHOD!!

        Method to recognize image food
        """

        return Dish(
            name=file.filename,
            ingredient_components=[],
            calories=250,
            carbohydrates=30.0,
            protein=5.0,
            fibers=3.1,
            fats=10,
            saturated_fats=0.3,
            sugars=2.4,
            added_sugars=0.4,
            LDL_cholesterol=3.2,
            VDL_cholesterol=1.0,
        )
