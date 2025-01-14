"""Main model serving API module."""
import os
import random
from base64 import decodebytes

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, InternalServerError

from .exceptions.exceptions import (UnsupportedContentTypeError,
                                    UserNotFoundError)
from .models import calorieLLM, recomendationsLLM
from .models.recognitionDLM import ImagePredictor
from .schemas.schemas import RequestLlm, ResponseDlm, ResponseLlm

# Load ENV file
load_dotenv()

# Instantiate FLASK app
app = Flask(__name__)

# Load FLASK and Custom settings from ENV
app.config['ENV'] = os.getenv('FLASK_ENV')
app.config['DEBUG'] = os.getenv('FLASK_DEBUG') == 'True'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH'))
app.config['ALLOWED_EXTENSIONS'] = set(
    os.getenv('ALLOWED_EXTENSIONS').split(',')
)

# Miscellaneous
def validate_content_type(validate_request):
    """
    Helper method for validating application content types.

    Args:
        validate_request: The Flask request object.

    Returns:
        The JSON body of the request if the content type is valid.

    Raises:
        UnsupportedContentTypeError: If the content type is not 'application/json'.
    """
    content_type = validate_request.headers.get('Content-Type')
    if content_type == 'application/json':
        return validate_request.json
    else:
        raise UnsupportedContentTypeError(
            f'Content-Type {content_type} is not supported!'
        )


# Error Handlers
@app.errorhandler(UnsupportedContentTypeError)
def handle_unsupported_content_type(error):
    """
    Handles the exception for unsupported content types.

    Args:
        error (UnsupportedContentTypeError): The exception raised when an unsupported content type is encountered.

    Returns:
        tuple: JSON response with the error message and HTTP status code 415 (Unsupported Media Type).
    """
    return jsonify({'error': str(error)}), 415


@app.errorhandler(UserNotFoundError)
def handle_user_not_found(error):
    """
    Handles the exception when a user is not found.

    Args:
        error (UserNotFoundError): The exception raised when a user is not found.

    Returns:
        tuple: JSON response with the error message and HTTP status code 404 (Not Found).
    """
    return jsonify({'error': str(error)}), 404


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    """
    Handles the exception for bad request.

    Args:
        error (BadRequest): The exception raised for a bad request.

    Returns:
        tuple: JSON response with the error message and HTTP status code 400 (Bad Request).
    """
    return jsonify({'error': 'Bad Request: ' + str(error)}), 400


@app.errorhandler(Exception)
def handle_generic_exception(error):
    """
    Handles generic exceptions that are not caught by other error handlers.

    Args:
        error (Exception): The exception that was raised.

    Returns:
        tuple: JSON response with a generic error message and HTTP status code 500 (Internal Server Error).
    """
    return (
        jsonify(
            {'error': 'An unexpected error occurred', 'details': str(error)}
        ),
        500,
    )


# Routes
@app.route('/intake_prediction', methods=['POST'])
def process_intake_prediction():
    """
    Processes a user’s biometric data to predict calorie intake.

    Args:
        request_json (dict): The JSON body of the request containing the `userID`.

    Returns:
        dict: A JSON response containing the predicted calorie intake.

    Raises:
        UserNotFoundError: If the user is not found in the system.
        UnsupportedContentTypeError: If the content type of the request is not 'application/json'.
        InternalServerError: If an unexpected error occurs during processing.
    """
    try:
        request_json = validate_content_type(request)
        llm_input = RequestLlm(userID=request_json['userID'])

        intake_prediction = calorieLLM.calculate_calories(llm_input.userID)
        return ResponseLlm(
            calorie_prediction=intake_prediction
        ).model_dump_json()
    except (UserNotFoundError, UnsupportedContentTypeError) as e:
        raise e
    except Exception as e:
        raise InternalServerError(
            f'Error processing intake prediction: {str(e)}'
        ) from e


@app.route('/image_prediction', methods=['POST'])
def process_image_prediction():
    """
    Processes an uploaded image for prediction of food class.

    Args:
        request (Flask Request): The Flask request object containing the image data in binary format.

    Returns:
        dict: A JSON response containing the predicted class, confidence, and all predictions.

    Raises:
        BadRequest: If no image file is found in the request.
        InternalServerError: If an unexpected error occurs during image processing.
    """
    file_path = ''
    try:
        if not request.data:
            raise BadRequest('No image file found in the request')

        filename = ''
        filename += str(random.randrange(0, 100))
        filename += '.jpg'

        while os.path.exists(
            '/'.join([app.config['UPLOAD_FOLDER'], filename])
        ):
            filename = ''
            filename += str(random.randrange(0, 100))
            filename += '.jpg'

        # Ensure the directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the uploaded file
        with open(
            '/'.join([app.config['UPLOAD_FOLDER'], filename]), 'wb'
        ) as f:
            f.write(decodebytes(request.data))

        # Secure the filename and save the file temporarily
        file_path = '/'.join([app.config['UPLOAD_FOLDER'], filename])

        # Use the ImagePredictor to make a prediction
        image_model = ImagePredictor()
        prediction = image_model.predict_image(file_path)

        image_output = ResponseDlm(
            predicted_class=prediction.get('predicted_class'),
            confidence=prediction.get('confidence'),
            all_predictions=prediction.get('all_predictions'),
        )

        return image_output.model_dump_json(), 200

    finally:
        # Ensure the file is removed, even if an exception occurs
        if os.path.exists(file_path):
            os.remove(file_path)


@app.route('/chat', methods=['POST'])
def nutri_chat():
    """
    Processes a given prompt from user to the chat-bot LLM and returns its recommendations.

    Returns:
        dict: A JSON response containing the LLM's suggestions.
    """
    data = request.get_json()

    response = recomendationsLLM.chat(data)
    return jsonify({'response': response})
