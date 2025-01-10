import base64
import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest

from app.app import app
from app.exceptions.exceptions import (UnsupportedContentTypeError,
                                       UserNotFoundError)


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    with app.test_client() as client:
        yield client


# Mock for calorieLLM
@patch('app.models.calorieLLM.calculate_calories', return_value=2000)
def test_process_intake_prediction(mock_calculate_calories, client):
    payload = {'userID': 0}

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


def test_process_intake_prediction_user_not_found(client):
    with patch(
        'app.models.calorieLLM.calculate_calories',
        side_effect=UserNotFoundError('User not found'),
    ):
        payload = {'userID': 0}

        response = client.post('/intake_prediction', json=payload)

        assert response.status_code == 404
        assert response.get_json()['error'] == 'User not found'


# @patch(
#     'app.models.recognitionDLM.ImagePredictor'
# )  # Mock the entire ImagePredictor class
# @patch('builtins.open')  # Mock file handling
# @patch('os.makedirs')  # Mock directory creation
# @patch('os.path.exists', return_value=False)  # Mock file existence check
# def test_process_image_prediction(
#     mock_exists, mock_makedirs, mock_open, mock_image_predictor, client
# ):
#     # Mock the ImagePredictor instance and its predict_image method
#     mock_image_predictor.return_value = MagicMock()
#     mock_image_predictor.predict_image.return_value = {
#         'predicted_class': 'cat',
#         'confidence': 0.95,
#         'all_predictions': [
#             ('cat', 0.95),
#             ('dog', 0.05),
#         ],
#     }

#     # Create a fake image file and encode it as Base64
#     fake_image_data = b'fake_binary_data'  # Simulate binary image content
#     fake_image_base64 = base64.b64encode(fake_image_data)

#     # Simulate sending a POST request with Base64-encoded data
#     response = client.post(
#         '/image_prediction',
#         data=fake_image_base64,  # Send valid Base64-encoded data
#     )

#     mock_open.return_value = {}
#     mock_exists.return_value = {}
#     mock_makedirs.return_value = {}

#     # Assert the response is successful and contains the expected prediction
#     assert response.data == 'u'
#     assert response.status_code == 200
#     assert response.get_json() == {
#         'predicted_class': 'cat',
#         'confidence': 0.95,
#         'all_predictions': [
#             ('cat', 0.95),
#             ('dog', 0.05),
#         ],
#     }


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
