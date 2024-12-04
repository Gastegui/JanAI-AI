"""Main model serving API module"""
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest, InternalServerError
from werkzeug.utils import secure_filename

from .exceptions.exceptions import (UnsupportedContentTypeError,
                                    UserNotFoundError)
from .models import calorieLLM

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
    elif content_type == 'application/xml':
        raise UnsupportedContentTypeError(
            f'Content-Type {content_type} is not supported!'
        )
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
        if not request_json or 'userID' not in request_json:
            raise BadRequest('Missing required field: userID')

        intake_prediction = calorieLLM.calculate_calories(
            request_json['userID']
        )
        return jsonify({'calorie_prediction': intake_prediction})
    except UserNotFoundError as e:
        raise e
    except Exception as e:
        raise InternalServerError(
            f'Error processing intake prediction: {str(e)}'
        ) from e


# @app.route('/image_prediction', methods=['POST'])
# def process_image_prediction():
#     """
#     Method to upload an image for prediction.
#     Returns a generic response for now.
#     """
#     try:
#         if 'img' not in request.files:
#             raise BadRequest('No image file found in the request')

#         file = request.files['img']
#         if file.filename == '':
#             raise BadRequest('No selected file')

#         if not allowed_file(file.filename):
#             raise BadRequest('File type not allowed')

#         # Secure the filename and save the file temporarily
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # Mock prediction result (replace with actual model logic later)
#         image_model = model.ImageModel()
#         mock_prediction = image_model.predict(file)

#         return jsonify(mock_prediction.model_dump()), 200
#     except Exception as e:
#         raise Exception(f"Error processing image prediction: {str(e)}")
