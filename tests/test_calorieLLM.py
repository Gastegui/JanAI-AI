"""Calorie LLM file test module"""
from unittest.mock import patch

from app.models.calorieLLM import calculate_calories

# Tests for CalorieLLM model file


@patch('langchain.chains.LLMChain.invoke')
@patch('mysql.connector')
def test_calculate_calories(mock_connector, mock_invoke):
    """
    Test case for the inner functioning of calorie calculation script.

    Args:
        mock_connector (MagicMock): Mock for the calorie LLM's response.
        mock_invoke (MagicMock): Mock for the MySQL db connection.

    Asserts:
        - 200 status code.
        - Proper return of chat response.
    """
    mock_invoke.return_value = {'text': '2000.00'}
    mock_connector.return_value = {}
    result = calculate_calories(1)
    assert result == '2000.00'


@patch('app.models.calorieLLM.get_user_data')
@patch('mysql.connector')
def test_calculate_calories_no_user(mock_connector, mock_get_user_data):
    """
    Test case for the inner functioning of calorie calculation script.

    Args:
        mock_connector (MagicMock): Mock for the calorie LLM's response.
        mock_get_user_data (MagicMock): Mock for the missing userData.

    Asserts:
        - 200 status code.
        - Proper return of chat response.
    """
    mock_get_user_data.return_value = None
    mock_connector.return_value = {}
    result = calculate_calories(1)
    assert result[0]['error'] == 'No user found with ID 1'
    assert result[1] == 404
