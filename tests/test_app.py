import base64
import io
import json
import math
import os
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.app import app
from app.exceptions.exceptions import (
    UnsupportedContentTypeError,
    UserNotFoundError,
)


@pytest.fixture
def client():
    with patch('mysql.connector.connect'):
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = '/upload'
        app.config['MAX_CONTENT_LENGTH'] = 16777216
        with app.test_client() as client:
            yield client


# Mock for calorieLLM
@patch('mysql.connector')
@patch('app.models.calorieLLM.calculate_calories', return_value=2000)
def test_process_intake_prediction(
    mock_calculate_calories, mock_connect, client
):
    payload = {'userID': 0}

    mock_cursor = MagicMock()
    mock_cursor.configure_mock(**{'fetchone.return_value': []})
    mock_connect.return_value = {}

    response = client.post('/intake_prediction', json=payload)

    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['calorie_prediction'] == 2000


def test_process_intake_prediction_missing_content_type(client):
    response = client.post('/intake_prediction', data='invalid payload')

    assert response.status_code == 415
    assert (
        json.loads(response.data)['error']
        == 'Content-Type None is not supported!'
    )


def test_process_intake_prediction_unexpected(client):
    with patch(
        'app.models.calorieLLM.calculate_calories',
        side_effect=Exception('Wildcard Exception'),
    ):
        payload = {'userID': 0}

        response = client.post('/intake_prediction', json=payload)

        assert response.status_code == 500
        assert response.get_json()['error'] == 'An unexpected error occurred'


def test_process_intake_prediction_user_not_found(client):
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
    # Mock the prediction response
    mock_image_predictor.return_value.predict_image.return_value = {
        'predicted_class': 'tortilla',
        'confidence': 0.95,
        'all_predictions': [
            ('tortilla', 0.95),
            ('pancakes', 0.03),
            ('waffles', 0.02),
        ],
    }

    # Create a fake image file (binary content)
    fake_image = io.BytesIO()
    fake_image.write(b'fake_image_content')
    fake_image.seek(0)

    # Encode the fake image as a base64 string
    encoded_image = base64.b64encode(fake_image.getvalue()).decode('utf-8')

    # Send the POST request with the base64 image as data
    response = client.post(
        '/image_prediction',
        data=base64.b64decode(encoded_image),  # Send binary data
    )

    # Assert the response
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['predicted_class'] == 'tortilla'
    assert math.isclose(response_data['confidence'], 0.95, rel_tol=0.2)
    assert len(response_data['all_predictions']) == 3

    # Assert that os.makedirs was called (to create the upload folder)
    mock_makedirs.assert_called_once_with('/upload', exist_ok=True)


def test_process_image_prediction_no_data(client):

    with patch('builtins.open', new_callable=MagicMock):
        response = client.post('/image_prediction')

    assert response.status_code == 400
    assert (
        response.get_json()['error']
        == 'Bad Request: 400 Bad Request: No image file found in the request'
    )


# Test Error Handlers
def test_handle_unsupported_content_type(client):
    with client.application.test_request_context():
        error = UnsupportedContentTypeError('Unsupported content type')
        response = client.application.handle_user_exception(error)

        assert response[1] == 415
        assert response[0].get_json()['error'] == 'Unsupported content type'


def test_handle_user_not_found(client):
    with client.application.test_request_context():
        error = UserNotFoundError('User not found')
        response = client.application.handle_user_exception(error)

        assert response[1] == 404
        assert response[0].get_json()['error'] == 'User not found'


def test_handle_generic_exception(client):
    with client.application.test_request_context():
        error = Exception('Something went wrong')
        response = client.application.handle_user_exception(error)

        assert response[1] == 500
        assert (
            response[0].get_json()['error'] == 'An unexpected error occurred'
        )
        assert response[0].get_json()['details'] == 'Something went wrong'
