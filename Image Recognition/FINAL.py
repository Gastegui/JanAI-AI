
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time

print(f"Cuda available: {torch.cuda.is_available()}")


# Configuration of hyperparams and other variables that will be used later
print("Autoencoder + classifier")
try:
    
    img_size = 256

    # Autoencoder stuff
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 10
    dropout = 0.1
    convolutional_kernel = 4
    convolutional_stride = 2
    max_pool_kernel = 2
    amount_of_pictures_to_show = 10
    encoder_name = "models/encoder_FINAL.pth"
    decoder_name = "models/decoder_FINAL.pth"
    autoencoder_name = "models/autoencoder_FINAL.pth"

    # Classificator stuff
    classification_batch_size = 128
    classification_learning_rate = 1e-3
    classification_epochs = 20
    encoder_name_to_load = encoder_name
    classifier_name = "models/classifier_FINAL.pth"



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    workers = os.cpu_count() - 2
    print(f"Using {workers} workers for loading the datasets")

    # # Autoencoder
    # 
    # Encodes the images first and then it decodes them, so it learns the most important patterns needed to recreate the image

    
    class ResidualEncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ResidualEncoderBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=dropout)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=convolutional_kernel, stride=convolutional_stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=dropout)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

            self.adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=convolutional_stride, bias=False) if in_channels != out_channels else None

        def forward(self, x):
            shorcut = x
            if self.adjust:
                shorcut = self.adjust(x)

            out = self.bn1(x)
            out = self.relu1(out)
            out = self.dropout1(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.dropout2(out)
            out = self.conv2(out)

            out =  out + shorcut
            return out

    class ResidualEncoder(nn.Module):
        def __init__(self):
            super(ResidualEncoder, self).__init__()
            self.block1 = ResidualEncoderBlock(3, 64)
            self.pool1 = nn.MaxPool2d(max_pool_kernel)
            self.block2 = ResidualEncoderBlock(64, 128)
            self.pool2 = nn.MaxPool2d(max_pool_kernel)
            self.block3 = ResidualEncoderBlock(128, 256)
        
    class ResidualDecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ResidualDecoderBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=dropout)
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=convolutional_kernel, stride=convolutional_stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=dropout)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

            self.adjust = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=convolutional_stride, output_padding=1, bias=False) if in_channels != out_channels else None

        def forward(self, x):
            shortcut = x
            if self.adjust:
                shortcut = self.adjust(x)

            out = self.bn1(x)
            out = self.relu1(out)
            out = self.dropout1(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.dropout2(out)
            out = self.conv2(out)
            
            out = out + shortcut
            return out


    class ResidualDecoder(nn.Module):
        def __init__(self):
            super(ResidualDecoder, self).__init__()
            self.block1 = ResidualDecoderBlock(256, 128)
            self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.block2 = ResidualDecoderBlock(128*2, 64) # *2 as it has the concat of the skip connection
            self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.block3 = ResidualDecoderBlock(64*2, 3) # same


    class ResidualAutoencoder(nn.Module):
        def __init__(self):
            super(ResidualAutoencoder, self).__init__()
            self.encoder = ResidualEncoder()
            self.decoder = ResidualDecoder()
            self.classi = Classifier()

        def forward(self, x):
            skip_connections = [None, None]
            
            # ENCODER

            x = self.encoder.block1(x)
            skip_connections[0] = x
            x = self.encoder.pool1(x)

            x = self.encoder.block2(x)
            skip_connections[1] = x
            x = self.encoder.pool2(x)

            latent = self.encoder.block3(x)

            # DECODER

            x = self.decoder.block1(latent)
            x = self.decoder.up1(x)
            
            x = torch.cat([x, skip_connections[1]], dim=1)
            x = self.decoder.block2(x)
            x = self.decoder.up2(x)

            x = torch.cat([x, skip_connections[0]], dim=1)
            reconstructed = self.decoder.block3(x)

            return latent, reconstructed

    # # Classifier
    # 
    # Uses the encoder from the autoencoder to "get" the important patterns of the image, and uses them for classifiying them

    
    class Classifier(nn.Module):
        def __init__(self, encoder):
            super(Classifier, self).__init__()
            self.encoder = encoder
            for param in self.encoder.parameters():
                param.requires_grad = True

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(256*8*8, 256*8) # 16384 -> 2048
            self.bn1 = nn.BatchNorm1d(256 * 8)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(p=dropout)
            self.fc2 = nn.Linear(256*8, 512) # 2048 -> 512
            self.bn2 = nn.BatchNorm1d(512)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(p=dropout)
            self.fc3 = nn.Linear(512, 101)
        def forward(self, x):
            x = self.encoder(x)
            x = self.flatten(x)

            x = self.fc1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.dropout1(x)
            
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.dropout2(x)

            x = self.fc3(x)
            return x

    # # Loading of the datasets

    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = ImageFolder('./Datasets/Cleaned/train', transform=transform_train)
    test_dataset = ImageFolder('./Datasets/Cleaned/test', transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    dataloaders = {"train": train_loader, "test": test_loader}

    
    # Inicialización del modelo, pérdida y optimizador
    model = ResidualAutoencoder()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    print(f"Device for model: {next(model.parameters()).device}")

    # ### Autoencoder training

    
    train_loss = []
    validation_loss = []

    since = time.time()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Modo de entrenamiento
            else:
                model.eval()  # Modo de evaluación

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            with tqdm(total=len(dataloaders[phase]), desc=f"{phase} phase") as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        _, outputs = model(inputs)
                        loss = criterion(outputs, inputs)

                        # Backward pass y optimización solo en entrenamiento
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)
                    
                    pbar.set_postfix({"Loss": f"{running_loss / total_samples:.4f}"})
                    pbar.update(1)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == "train":
                train_loss.append(float(epoch_loss))
            else:
                validation_loss.append(float(epoch_loss))

            # Sincronizar GPU para evitar acumulación de operaciones pendientes
            torch.cuda.synchronize()

        time_elapsed = time.time() - since
        print(f'Epoch finished at {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # Guardar el encoder para su uso posterior
    torch.save(model.encoder.state_dict(), encoder_name)
    torch.save(model.decoder.state_dict(), decoder_name)
    torch.save(model.state_dict(), autoencoder_name)

    plt.plot(range(len(train_loss)), train_loss, label="Train")
    plt.plot(range(len(validation_loss)), validation_loss, label="Validation")
    plt.title("Loss")
    plt.legend("upper right")
    plt.show()

    # # Training of the classification model

    
    pretrained_encoder = ResidualAutoencoder().encoder
    pretrained_encoder.load_state_dict(torch.load(encoder_name_to_load))
    pretrained_encoder = pretrained_encoder.to(device)

    model = Classifier(pretrained_encoder).to(device)

    train_loader = DataLoader(train_dataset, batch_size=classification_batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=classification_batch_size, shuffle=False, num_workers=workers)
    dataloaders = {"train": train_loader, "test": test_loader}


    
    def train_model(model, criterion, optimizer, scheduler, num_epochs):
        since = time.time()
        train_acc = []
        train_loss = []
        validation_acc = []
        validation_loss = []
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                with tqdm(total=len(dataloaders[phase]), desc=f"{phase} phase") as pbar:
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        batch_acc = (torch.sum(preds == labels.data).item() / inputs.size(0)) * 100
                        pbar.set_postfix({"Loss": running_loss, "Accuracy": f"{batch_acc:.2f}%"})
                        pbar.update(1)
                        
                if phase == 'train':
                    scheduler.step(epoch_loss)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                if phase == "train":
                    train_acc.append(float(epoch_acc))
                    train_loss.append(float(epoch_loss))
                else:
                    validation_acc.append(float(epoch_acc))
                    validation_loss.append(float(epoch_loss))

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            print()

            time_elapsed = time.time() - since
            print(f'Epoch finished at {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        torch.save(model.state_dict(), classifier_name)

        plt.subplot(1, 2, 1)
        plt.plot(range(len(train_acc)), train_acc, label="Train")
        plt.plot(range(len(validation_acc)), validation_acc, label="Validation")
        plt.legend(loc="lower right")
        plt.title("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(range(len(train_loss)), train_loss, label="Train")
        plt.plot(range(len(validation_loss)), validation_loss, label="Validation")
        plt.legend("lower right")
        plt.title("Loss")

        plt.show()

        return model

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=classification_learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    
    model = train_model(model, criterion, optimizer, scheduler, classification_epochs)
except KeyboardInterrupt:
    print("Interrupted")
except Exception as e:
    print("Other exception: " + str(e))
    raise e