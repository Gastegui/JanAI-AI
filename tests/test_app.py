import base64
import io
import json
import math
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.app import app
from app.exceptions.exceptions import (
    UnsupportedContentTypeError,
    UserNotFoundError,
)


@pytest.fixture
def client():
    """Fixture for initializing the Flask test client.

    Returns:
        flask.testing.FlaskClient: The test client for making requests.
    """
    with patch('mysql.connector.connect'):
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = '/upload'
        app.config['MAX_CONTENT_LENGTH'] = 16777216
        with app.test_client() as client:
            yield client


# Mock for calorieLLM
@patch('app.models.calorieLLM.calculate_calories', return_value=2000)
def test_process_intake_prediction(mock_calculate_calories, client):
    """
    Test case for processing the intake prediction.

    Args:
        mock_calculate_calories (MagicMock): Mock for the calorie calculation function.
        mock_connect (MagicMock): Mock for the database connection.
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 200 status code.
        - Calorie prediction in the response.
    """
    payload = {'userID': 0}

    response = client.post('/intake_prediction', json=payload)

    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['calorie_prediction'] == 2000


def test_process_intake_prediction_missing_content_type(client):
    """
    Test case for handling missing content type in the request.

    Args:
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 415 status code.
        - Error message regarding missing content type.
    """
    response = client.post('/intake_prediction', data='invalid payload')

    assert response.status_code == 415
    assert (
        json.loads(response.data)['error']
        == 'Content-Type None is not supported!'
    )


def test_process_intake_prediction_unexpected(client):
    """
    Test case for handling unexpected errors during intake prediction.

    Args:
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 500 status code.
        - Error message indicating an unexpected error.
    """
    with patch(
        'app.models.calorieLLM.calculate_calories',
        side_effect=Exception('Wildcard Exception'),
    ):
        payload = {'userID': 0}

        response = client.post('/intake_prediction', json=payload)

        assert response.status_code == 500
        assert response.get_json()['error'] == 'An unexpected error occurred'


def test_process_intake_prediction_user_not_found(client):
    """
    Test case for handling user not found errors.

    Args:
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 404 status code.
        - Error message for user not found.
    """
    with patch(
        'app.models.calorieLLM.calculate_calories',
        side_effect=UserNotFoundError('User not found'),
    ):
        payload = {'userID': 0}

        response = client.post('/intake_prediction', json=payload)

        assert response.status_code == 404
        assert response.get_json()['error'] == 'User not found'


@patch('app.app.ImagePredictor')
@patch('app.app.os.makedirs')
@patch('app.app.open', new_callable=mock_open)
def test_process_image_prediction_success(
    mock_open, mock_makedirs, mock_image_predictor, client
):
    """
    Test case for successful image prediction.

    Args:
        mock_open (MagicMock): Mock for opening files.
        mock_makedirs (MagicMock): Mock for creating directories.
        mock_image_predictor (MagicMock): Mock for the image prediction function.
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 200 status code.
        - Correct response data including predictions.
    """
    mock_image_predictor.return_value.predict_image.return_value = {
        'predicted_class': 'tortilla',
        'confidence': 0.95,
        'all_predictions': [
            ('tortilla', 0.95),
            ('pancakes', 0.03),
            ('waffles', 0.02),
        ],
    }

    fake_image = io.BytesIO()
    fake_image.write(b'fake_image_content')
    fake_image.seek(0)

    encoded_image = base64.b64encode(fake_image.getvalue()).decode('utf-8')

    response = client.post(
        '/image_prediction',
        data=base64.b64decode(encoded_image),
    )

    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['predicted_class'] == 'tortilla'
    assert math.isclose(response_data['confidence'], 0.95, rel_tol=0.2)
    assert len(response_data['all_predictions']) == 3

    mock_makedirs.assert_called_once_with('/upload', exist_ok=True)


def test_process_image_prediction_no_data(client):
    """
    Test case for handling requests with no image data.

    Args:
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 400 status code.
        - Error message indicating missing image data.
    """
    with patch('builtins.open', new_callable=MagicMock):
        response = client.post('/image_prediction')

    assert response.status_code == 400
    assert (
        response.get_json()['error']
        == 'Bad Request: 400 Bad Request: No image file found in the request'
    )


# Mock for recommendationLLM
@patch('app.models.recomendationsLLM.chat', return_value='Control yourself')
def test_process_chat(mock_recommendation, client):
    """
    Test case for processing the chat bot.

    Args:
        mock_recommendation (MagicMock): Mock for the recommendation LLM's response.
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 200 status code.
        - Proper return of chat response.
    """
    payload = {
        'context': 'Id like to have a cheat meal today, what can i eat',
        'username': 'lukeniri',
    }

    mock_recommendation.return_value = 'Control yourself, have a snickers'

    response = client.post('/chat', json=payload)

    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data is not None
    assert response_data['response'] == 'Control yourself, have a snickers'


# Test Error Handlers


def test_handle_unsupported_content_type(client):
    """
    Test case for handling unsupported content type.

    Args:
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 415 status code.
        - Error message for unsupported content type.
    """
    with client.application.test_request_context():
        error = UnsupportedContentTypeError('Unsupported content type')
        response = client.application.handle_user_exception(error)

        assert response[1] == 415
        assert response[0].get_json()['error'] == 'Unsupported content type'


def test_handle_user_not_found(client):
    """
    Test case for handling user not found errors.

    Args:
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 404 status code.
        - Error message for user not found.
    """
    with client.application.test_request_context():
        error = UserNotFoundError('User not found')
        response = client.application.handle_user_exception(error)

        assert response[1] == 404
        assert response[0].get_json()['error'] == 'User not found'


def test_handle_generic_exception(client):
    """
    Test case for handling generic exceptions.

    Args:
        client (flask.testing.FlaskClient): The test client for making requests.

    Asserts:
        - 500 status code.
        - Error message for unexpected errors.
    """
    with client.application.test_request_context():
        error = Exception('Something went wrong')
        response = client.application.handle_user_exception(error)

        assert response[1] == 500
        assert (
            response[0].get_json()['error'] == 'An unexpected error occurred'
        )
        assert response[0].get_json()['details'] == 'Something went wrong'
