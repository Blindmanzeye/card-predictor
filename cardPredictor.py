"""Card Predictor: train a neural network to identify a playing card in an image.

Paths are resolved relative to this file (not the current working directory),
so the script can be run from anywhere:

    project root/
        cardPredictor.py
        resources/
            train/<card class>/*.png
            valid/<card class>/*.png
            test/<card class>/*.png
            cards.csv
            *.h5            (earlier model experiments — see resources/README.md)

The .h5 files in resources/ are artefacts of earlier experiments. This script
trains a PyTorch model and does not load them.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt  # For data viz
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

print('System Version:', sys.version)
print('PyTorch version', torch.__version__)
print('Torchvision version', torchvision.__version__)
print('Numpy version', np.__version__)
print('Pandas version', pd.__version__)


# --- Project paths (relative to this file, never the working directory) ---
BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
TRAIN_DIR = RESOURCES_DIR / "train"
VALID_DIR = RESOURCES_DIR / "valid"
TEST_DIR = RESOURCES_DIR / "test"
CARDS_CSV = RESOURCES_DIR / "cards.csv"
# Earlier HDF5 model experiments kept in resources/ (not loaded by this script).
LEGACY_MODEL_FILES = (
    RESOURCES_DIR / "14card types - 14-(200 x 200)-94.61.h5",
    RESOURCES_DIR / "53cards-53-(200x200)-100.00.h5",
)


class PlayingCardDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(str(data_dir), transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes


class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super().__init__()
        # Where we define all the parts of the model
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        enet_out_size = 1280
        # Make a classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        # Connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output


# Backwards-compatible alias: the original (misspelled) class name.
SimpleCardClassifer = SimpleCardClassifier


def validate_dataset_layout():
    """Fail early, with a clear message, if resources/ is not laid out as
    ImageFolder expects: one subfolder per card class inside each split,
    and no image files sitting directly in a split folder."""
    problems = []
    for split_dir in (TRAIN_DIR, VALID_DIR, TEST_DIR):
        if not split_dir.is_dir():
            problems.append(f"Missing dataset folder: {split_dir}")
            continue
        stray_files = [p.name for p in split_dir.iterdir() if p.is_file() and p.name != ".gitkeep"]
        if stray_files:
            problems.append(
                f"{split_dir.name}/ contains files directly ({', '.join(sorted(stray_files))}); "
                "images must be inside a subfolder named after their card class."
            )
        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            problems.append(
                f"{split_dir.name}/ has no class subfolders. Expected one folder per card "
                "(e.g. 'ace of spades'), each containing that card's images."
            )
    if problems:
        raise SystemExit(
            "Dataset layout problem — the project cannot train yet:\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\nSee resources/README.md for the expected layout."
        )


# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


# Visualization
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


def main():
    validate_dataset_layout()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = PlayingCardDataset(TRAIN_DIR, transform=transform)
    val_dataset = PlayingCardDataset(VALID_DIR, transform=transform)
    test_dataset = PlayingCardDataset(TEST_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Simple training loop
    num_epochs = 5
    train_losses, val_losses = [], []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimpleCardClassifier(num_classes=len(train_dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    class_names = train_dataset.classes

    while True:
        test_image = input("Please Input the file path of the image or (n): ")
        if test_image.lower() == "n":
            break

        original_image, image_tensor = preprocess_image(test_image, transform)
        probabilities = predict(model, image_tensor, device)
        visualize_predictions(original_image, probabilities, class_names)


if __name__ == "__main__":
    main()
