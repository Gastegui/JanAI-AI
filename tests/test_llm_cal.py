from unittest.mock import MagicMock, patch

import pytest

from app.models.calorieLLM import calculate_calories


@patch('app.models.calorieLLM.getUserData')
@patch('app.models.calorieLLM.getWeightData')
@patch('langchain.chains.LLMChain.invoke')
def test_calculate_calories(mock_invoke, mock_getWeightData, mock_getUserData):
    mock_getUserData.return_value = {
        'height': 175,
        'age': 30,
        'waist': 80,
        'neck': 40,
        'gender': 'male',
        'activityLevel': 'Moderately Active',
        'bmrMifflin': 1500,
        'bmrHarrisBenedict': 1550,
        'bmrKatchMcArdle': 1450,
        'tdeeMifflin': 2500,
        'tdeeHarrisBenedict': 2600,
        'tdeeKatchMcArdle': 2400,
        'bodyFat': 20,
        'totalWeightLoss': 10,
        'weeklyDeficit': 7000,
        'dailyCalorieIntakeMifflin': 2000,
        'dailyCalorieIntakeHarrisBenedict': 2100,
        'dailyCalorieIntakeKatchMcArdle': 1900,
        'objective': 'Lose weight',
    }
    mock_getWeightData.return_value = {
        'weight': 80,
        'goalWeight': 70,
        'durationToAchieveGoalWeight': 12,
    }
    mock_invoke.return_value = {'text': '2000.00'}

    result = calculate_calories(1)
    assert result == '2000.00'
