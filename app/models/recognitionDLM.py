"""
Module file dedicated to storing model classes
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image

IMG_SIZE = 224


def preprocess(org_path: str, train: bool):
    """
    Method for preprocessing ...
    """
    arr = org_path.split('/')
    dst_start = '/'.join(arr[: arr.index('Datasets') + 1])
    dst = ''
    if train:
        dst = (
            dst_start + '/Cleaned/train/' + '/'.join(org_path.split('/')[-2:])
        )
    else:
        dst = dst_start + '/Cleaned/test/' + '/'.join(org_path.split('/')[-2:])

    if not (dst.endswith('.jpg') or dst.endswith('.jpeg')):
        dst = '.'.join(dst.split('.')[:-1]) + '.jpg'

    if os.path.exists(dst):
        print('File already preprocessed')
        return

    image = Image.open(org_path)
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.HAMMING)

    if image.mode in (
        'RGBA',
        'LA',
    ):   # If the image has transparency, get rid of it
        background = Image.new(
            'RGB', image.size, (255, 255, 255)
        )   # Create a white image to act as the background
        background.paste(
            image, mask=image.split()[3]
        )   # Apply this background where there is transparency on the image
        image = background

    os.makedirs('/'.join(dst.split('/')[:-1]), exist_ok=True)
    image.save(dst, 'JPEG')


class Autoencoder(nn.Module):
    """
    Class for ...
    """

    def __init__(self):
        super(Autoencoder, self).__init__()

        # Autoencoder stuff
        dropout = 0.3
        convolutional_kernel = 4

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3, 64, kernel_size=convolutional_kernel, stride=2, padding=1
            ),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=max_pool_kernel),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(
                64, 128, kernel_size=convolutional_kernel, stride=2, padding=1
            ),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=max_pool_kernel),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(
                128, 256, kernel_size=convolutional_kernel, stride=2, padding=1
            ),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=max_pool_kernel),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(
                256, 512, kernel_size=convolutional_kernel, stride=2, padding=1
            ),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=max_pool_kernel),
            nn.Dropout2d(p=dropout),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=convolutional_kernel, stride=2, padding=1
            ),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.ConvTranspose2d(
                256, 128, kernel_size=convolutional_kernel, stride=2, padding=1
            ),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.ConvTranspose2d(
                128, 64, kernel_size=convolutional_kernel, stride=2, padding=1
            ),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.ConvTranspose2d(
                64, 3, kernel_size=convolutional_kernel, stride=2, padding=1
            ),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid(),  # Normalización de salida entre 0 y 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Classifier(nn.Module):
    def __init__(self, encoder):
        super(Classifier, self).__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder[12].parameters():
            param.requires_grad = True

        for param in self.encoder[13].parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(512 * 14 * 14, 512),  # Ajusta según la salida del encoder
            # nn.ReLU(),
            nn.Linear(
                512 * 14 * 14, 101
            ),  # Supongamos que tienes 101 clases de platos
        )

    def forward(self, x):
        x = self.encoder(x)  # Congela el encoder si ya está preentrenado
        x = self.fc(x)
        return x


class ImagePredictor:
    def __init__(self):
        # Initialize a new model instance
        self.encoder = Autoencoder().encoder

        print('1')

        self.encoder.load_state_dict(
            torch.load(os.getenv('MODEL_ENCODER_PATH'), weights_only=True)
        )
        self.model = Classifier(encoder=self.encoder)
        # Load the saved state dict

        print('2')

        self.model.load_state_dict(
            torch.load(os.getenv('MODEL_PATH'), weights_only=True)
        )
        print('3')
        self.model.eval()  # Set to evaluation mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # Store class names
        self.class_names = [
            'apple_pie',
            'baby_back_ribs',
            'baklava',
            'beef_carpaccio',
            'beef_tartare',
            'beet_salad',
            'beignets',
            'bibimbap',
            'bread_pudding',
            'breakfast_burrito',
            'bruschetta',
            'caesar_salad',
            'cannoli',
            'caprese_salad',
            'carrot_cake',
            'ceviche',
            'cheese_plate',
            'cheesecake',
            'chicken_curry',
            'chicken_quesadilla',
            'chicken_wings',
            'chocolate_cake',
            'chocolate_mousse',
            'churros',
            'clam_chowder',
            'club_sandwich',
            'crab_cakes',
            'creme_brulee',
            'croque_madame',
            'cup_cakes',
            'deviled_eggs',
            'donuts',
            'dumplings',
            'edamame',
            'eggs_benedict',
            'escargots',
            'falafel',
            'filet_mignon',
            'fish_and_chips',
            'foie_gras',
            'french_fries',
            'french_onion_soup',
            'french_toast',
            'fried_calamari',
            'fried_rice',
            'frozen_yogurt',
            'garlic_bread',
            'gnocchi',
            'greek_salad',
            'grilled_cheese_sandwich',
            'grilled_salmon',
            'guacamole',
            'gyoza',
            'hamburger',
            'hot_and_sour_soup',
            'hot_dog',
            'huevos_rancheros',
            'hummus',
            'ice_cream',
            'lasagna',
            'lobster_bisque',
            'lobster_roll_sandwich',
            'macaroni_and_cheese',
            'macarons',
            'miso_soup',
            'mussels',
            'nachos',
            'omelette',
            'onion_rings',
            'oysters',
            'pad_thai',
            'paella',
            'pancakes',
            'panna_cotta',
            'peking_duck',
            'pho',
            'pizza',
            'pork_chop',
            'poutine',
            'prime_rib',
            'pulled_pork_sandwich',
            'ramen',
            'ravioli',
            'red_velvet_cake',
            'risotto',
            'samosa',
            'sashimi',
            'scallops',
            'seaweed_salad',
            'shrimp_and_grits',
            'spaghetti_bolognese',
            'spaghetti_carbonara',
            'spring_rolls',
            'steak',
            'strawberry_shortcake',
            'sushi',
            'tacos',
            'takoyaki',
            'tiramisu',
            'tuna_tartare',
            'waffles',
        ]

        # Define the same transforms used during training
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def predict_image(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(
            0
        )  # Add batch dimension
        image_tensor = image_tensor.to(self.device)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()

            # Get all probabilities as a list
            all_probs = probabilities[0].cpu().numpy()

            # Get predicted class name
            predicted_class_name = self.class_names[predicted_class_idx]

            # Create a list of (class_name, probability) tuples
            class_probabilities = [
                (class_name, float(prob))
                for class_name, prob in zip(self.class_names, all_probs)
            ]
            # Sort by probability in descending order
            class_probabilities.sort(key=lambda x: x[1], reverse=True)
        return {
            'predicted_class': predicted_class_name,
            'confidence': float(confidence),
            'all_predictions': class_probabilities,
        }

    def predict_batch(self, image_paths):
        # Process multiple images at once
        batch_tensors = []
        for path in image_paths:
            image = Image.open(path).convert('RGB')
            tensor = self.transform(image)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_indices = torch.argmax(probabilities, dim=1).tolist()
            confidences = [
                probabilities[i][pred].item()
                for i, pred in enumerate(predicted_indices)
            ]

            # Get predicted class names
            predicted_classes = [
                self.class_names[idx] for idx in predicted_indices
            ]

        return list(zip(predicted_classes, confidences))
