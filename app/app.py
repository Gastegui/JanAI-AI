"""Main model serving API module"""
import os
import random
from base64 import decodebytes

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest, InternalServerError

from .exceptions.exceptions import (UnsupportedContentTypeError,
                                    UserNotFoundError)
from .models import calorieLLM
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


# Helper Methods
def validate_content_type(validate_request):
    """
    Helper method for validating application content types
    """
    content_type = validate_request.headers.get('Content-Type')
    if content_type == 'application/json':
        return validate_request.json
    else:
        raise UnsupportedContentTypeError(
            f'Content-Type {content_type} is not supported!'
        )


def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    """
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower()
        in app.config['ALLOWED_EXTENSIONS']
    )


# Error Handlers
@app.errorhandler(UnsupportedContentTypeError)
def handle_unsupported_content_type(error):
    """Error handler for unsupported content type"""
    return jsonify({'error': str(error)}), 415


@app.errorhandler(UserNotFoundError)
def handle_user_not_found(error):
    """Error handler for user not found"""
    return jsonify({'error': str(error)}), 404


@app.errorhandler(BadRequest)
def handle_bad_request(error):
    """Error handler for bad request"""
    return jsonify({'error': 'Bad Request: ' + str(error)}), 400


@app.errorhandler(Exception)
def handle_generic_exception(error):
    """Error handling for unknown exception"""
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
    Method to pass user biometric data to
    intake prediction model
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
    Method to upload an image for prediction.
    Returns a generic response for now.
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

        with open(
            '/'.join([app.config['UPLOAD_FOLDER'], filename]), 'wb'
        ) as f:
            f.write(decodebytes(request.data))

        # Secure the filename and save the file temporarily
        file_path = '/'.join([app.config['UPLOAD_FOLDER'], filename])

        # Mock prediction result (replace with actual model logic later)
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
