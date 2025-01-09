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

    assert response[1].status_code == 200
    response_data = response[0].get_json()
    assert response_data['calorie_prediction'] == 2000


def test_process_intake_prediction_missing_content_type(client):
    response = client.post('/intake_prediction', data='invalid payload')

    assert response[1].status_code == 415
    assert (
        response[0].get_json()['error']
        == 'Content-Type None is not supported!'
    )


def test_process_intake_prediction_user_not_found(client):
    with patch(
        'app.models.calorieLLM.calculate_calories',
        side_effect=UserNotFoundError('User not found'),
    ):
        payload = {'userID': 'non_existent_user'}

        response = client.post('/intake_prediction', json=payload)

        assert response[1].status_code == 404
        assert response[0].get_json()['error'] == 'User not found'


# Mock for ImagePredictor
@patch(
    'app.models.recognitionDLM.ImagePredictor.predict_image',
    return_value={'prediction': 'cat'},
)
def test_process_image_prediction(mock_predict_image, client):
    file_data = b'base64_encoded_image_data'

    with patch('builtins.open', new_callable=MagicMock):
        response = client.post('/image_prediction', data=file_data)

    assert response[1].status_code == 200
    assert response[0].get_json() == {'prediction': 'cat'}


@patch('os.remove')
@patch(
    'app.models.recognitionDLM.ImagePredictor.predict_image',
    return_value={'prediction': 'dog'},
)
def test_process_image_prediction_cleanup(
    mock_predict_image, mock_os_remove, client
):
    file_data = b'base64_encoded_image_data'

    with patch('builtins.open', new_callable=MagicMock):
        response = client.post('/image_prediction', data=file_data)

    assert response[1] == 200
    assert response[0].get_json() == {'prediction': 'dog'}
    mock_os_remove.assert_called_once()


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
