from unittest.mock import MagicMock, call, patch

import numpy as np
import torch
from PIL import Image
from pytest import approx
from torchvision.transforms import Compose

from app.models.recognitionDLM import (
    Autoencoder,
    Classifier,
    ImagePredictor,
    preprocess,
)


# Helper: Mock environment variables
@patch('os.getenv')
@patch('os.makedirs')
@patch('os.path.exists')
@patch('PIL.Image.Image.save')
@patch('PIL.Image.Image.paste')
@patch('PIL.Image.open')
def test_preprocess(
    mock_image_open,
    mock_image_paste,
    mock_image_save,
    mock_exists,
    mock_makedirs,
    mock_getenv,
):
    """
    Test the preprocess function that handles image loading, resizing,
    and saving the processed image. Mocks image operations like opening,
    resizing, and saving to ensure the expected behavior of preprocess.

    Args:
        mock_image_open: Mock for PIL.Image.open method.
        mock_image_paste: Mock for PIL.Image.paste method.
        mock_image_save: Mock for PIL.Image.save method.
        mock_exists: Mock for os.path.exists method.
        mock_makedirs: Mock for os.makedirs method.
        mock_getenv: Mock for os.getenv method.
    """
    mock_image = MagicMock()
    mock_image.resize.return_value = mock_image
    mock_image.mode = 'RGBA'
    mock_image.size = (224, 224)
    mock_image.split.return_value = [None, None, None, MagicMock()]
    mock_image_open.return_value = mock_image
    mock_image_paste.return_value = {}
    mock_image_save.return_value = {}
    mock_exists.return_value = False

    preprocess('path/to/Datasets/train/image.png', train=True)

    mock_image_open.assert_called_once_with('path/to/Datasets/train/image.png')
    mock_image.resize.assert_called_once_with(
        (224, 224), Image.Resampling.HAMMING
    )
    mock_makedirs.assert_called_once()


def test_autoencoder_forward_pass():
    """
    Test the forward pass of the Autoencoder model. Verifies that the
    output of the autoencoder has the correct shapes for the encoded
    and decoded tensors.

    Ensures that the encoded tensor has the expected shape, and that
    the decoded tensor matches the input shape.

    """
    model = Autoencoder()
    mock_input = torch.rand(
        (2, 3, 224, 224)
    )  # Batch of 2, 3 channels, 224x224 image
    encoded, decoded = model(mock_input)

    assert encoded.shape[1] == 512  # Ensure the final encoded size is correct
    assert (
        decoded.shape == mock_input.shape
    )  # Ensure the decoded output matches input shape


@patch('PIL.Image.open')
@patch('torch.nn.functional.softmax')
@patch.object(ImagePredictor, '__init__', return_value=None)
def test_predict_image(mock_init, mock_softmax, mock_image_open):
    """
    Test the predict_image function in the ImagePredictor class.
    This test ensures that the image loading, prediction, and confidence
    calculation work as expected. It checks that the predicted class
    and confidence values are correct based on the mock output.

    Args:
        mock_init: Mock for the constructor of ImagePredictor.
        mock_softmax: Mock for the softmax function used in predictions.
        mock_image_open: Mock for PIL.Image.open used to load images.
    """
    mock_image = MagicMock()
    mock_image.convert.return_value = mock_image
    mock_image_open.return_value = mock_image
    mock_softmax.return_value = torch.tensor(
        [[0.1] * 100 + [0.9]]
    )  # High prob on last class

    predictor = ImagePredictor()
    predictor.model = MagicMock()
    predictor.device = 'cpu'
    predictor.transform = MagicMock(return_value=torch.rand((3, 224, 224)))
    predictor.class_names = [f'class_{i}' for i in range(101)]

    result = predictor.predict_image('path/to/image.jpg')

    mock_image_open.assert_called_once_with('path/to/image.jpg')
    predictor.model.assert_called_once()
    assert result['predicted_class'] == 'class_100'
    assert np.isclose(a=result['confidence'], b=0.9)
    assert len(result['all_predictions']) == 101


@patch('PIL.Image.open')
@patch.object(ImagePredictor, '__init__', return_value=None)
def test_predict_batch(mock_init, mock_image_open):
    """
    Test the predict_batch function in the ImagePredictor class.
    This test ensures that the batch image predictions work as expected,
    validating the proper loading, transformation, and prediction of multiple
    images at once.

    Args:
        mock_init: Mock for the constructor of ImagePredictor.
        mock_image_open: Mock for PIL.Image.open used to load images.
    """
    mock_image = MagicMock()
    mock_image.convert.return_value = mock_image
    mock_image_open.return_value = mock_image

    predictor = ImagePredictor()
    predictor.model = MagicMock()
    predictor.device = 'cpu'
    predictor.transform = MagicMock(
        side_effect=lambda img: torch.rand((3, 224, 224))
    )
    predictor.class_names = [f'class_{i}' for i in range(101)]

    mock_output = torch.tensor(
        [[0.1] * 100 + [0.9], [0.05] * 50 + [0.95] * 51]
    )
    predictor.model.return_value = mock_output

    result = predictor.predict_batch(['img1.jpg', 'img2.jpg'])

    mock_image_open.assert_has_calls(
        [
            call('img1.jpg'),
            call().convert('RGB'),
            call('img2.jpg'),
            call().convert('RGB'),
        ]
    )
    assert result[0][0] == 'class_100'
    assert result[1][0] == 'class_50'
    assert np.isclose(a=result[0][1], b=0.02, atol=0.1)
    assert np.isclose(a=result[1][1], b=0.015, atol=0.1)


def test_classifier_forward_pass():
    """
    Test the forward pass of the Classifier model. Verifies that the
    classifier produces the correct output shape and calls the encoder
    with the appropriate input tensor.

    Ensures that the classifier output has the correct shape (batch size,
    number of classes) and that the encoder is called once with the input tensor.
    """
    mock_encoder = MagicMock()
    mock_encoder.return_value = torch.rand(
        (2, 512, 14, 14)
    )  # Simulate encoder output
    model = Classifier(encoder=mock_encoder)

    mock_input = torch.rand((2, 3, 224, 224))
    output = model(mock_input)

    assert output.shape == (2, 101)  # Batch size 2, 101 classes
    assert mock_encoder.called_once_with(mock_input)


@patch('PIL.Image.open')
@patch('torch.nn.functional.softmax')
@patch.object(ImagePredictor, '__init__', return_value=None)
def test_predict_image_class_probabilities(
    mock_init, mock_softmax, mock_image_open
):
    """
    Test the prediction of image class probabilities. This test ensures that
    the softmax function correctly computes the probabilities for each class,
    and the predicted class with the highest probability is selected.

    Args:
        mock_init: Mock for the constructor of ImagePredictor.
        mock_softmax: Mock for the softmax function used in predictions.
        mock_image_open: Mock for PIL.Image.open used to load images.
    """
    mock_softmax.return_value = torch.tensor(
        [[0.05] * 50 + [0.2] * 50 + [0.3]]
    )
    mock_image_open.return_value = MagicMock()
    predictor = ImagePredictor()
    predictor.class_names = [f'class_{i}' for i in range(101)]
    predictor.transform = MagicMock(return_value=torch.rand((3, 224, 224)))
    predictor.model = MagicMock(return_value=mock_softmax.return_value)
    predictor.device = 'cpu'

    result = predictor.predict_image('mock/image.jpg')

    assert (
        result['all_predictions'][-1][0] == 'class_49'
    )  # Highest confidence class
    assert np.isclose(
        a=result['all_predictions'][-1][1], b=0.1, atol=0.2
    )  # Highest confidence class
    assert len(result['all_predictions']) == 101  # Full list of probabilities
