# Card Predictor — Playing Card Recognition with Transfer Learning

A computer vision project in **Python / PyTorch** that trains a neural network to identify which playing card is shown in an image. The model outputs a probability for every card class, and the result is visualised next to the image so a human can see not just the prediction, but how confident the model is in each alternative.

> **For recruiters and hiring managers:** this is a compact, end-to-end machine learning project — dataset pipeline, transfer-learning model, training/validation loop, and interactive inference — written from scratch rather than assembled from a tutorial template. A skills summary is in [Skills demonstrated](#skills-demonstrated) below.

---

## What it does

1. Loads a dataset of playing-card images organised into `train`, `valid`, and `test` splits.
2. Fine-tunes an **EfficientNet-B0** model pre-trained on ImageNet, replacing its classification head with a 53-class output layer (52 cards + joker).
3. Trains with cross-entropy loss and the Adam optimiser, tracking training and validation loss per epoch.
4. Runs interactively: you supply the path to a card image, and it displays the image beside a probability bar chart for all classes (softmax confidences from 0 to 1).

## How it works

```
resources/train ─┐
resources/valid ─┼─► ImageFolder + custom Dataset ─► DataLoader (batch 32, shuffle train)
resources/test  ─┘            │
                              ▼
              Resize (128 × 128) + ToTensor
                              │
                              ▼
        EfficientNet-B0 features (timm, pre-trained) ─► Flatten ─► Linear (1280 ─► 53)
                              │
                    CrossEntropyLoss + Adam (lr 0.001), 5 epochs, CUDA if available
                              │
                              ▼
        Inference: softmax probabilities ─► Matplotlib chart (image + per-class bars)
```

Key implementation points, all in `cardPredictor.py`:

- **Custom dataset class** — `PlayingCardDataset` wraps torchvision's `ImageFolder`, exposing `__len__`, `__getitem__`, and a `classes` property.
- **Transfer learning** — the network reuses EfficientNet-B0's learned features (`timm.create_model('efficientnet_b0', pretrained=True)`) and trains only a new classifier head, rather than training a CNN from scratch on a small dataset.
- **Clean train/eval discipline** — `model.train()` / `model.eval()` with `torch.no_grad()` for validation and inference, and device-aware execution (`cuda:0` when available, otherwise CPU).
- **Interpretable output** — predictions are shown as a full probability distribution, not a single label, which makes misclassifications and near-misses easy to inspect.

## Tech stack

| Area | Used |
|---|---|
| Language | Python |
| Deep learning | PyTorch, torchvision, timm |
| Model | EfficientNet-B0 (pre-trained), 53-class classifier head |
| Data | Custom `Dataset` / `DataLoader` pipeline over `ImageFolder` splits |
| Data handling | NumPy, Pandas, Pillow |
| Visualisation | Matplotlib |
| Concepts | Transfer learning, image classification, train/validation splits, softmax confidence, loss tracking |

## Skills demonstrated

- Building an end-to-end image-classification pipeline: ingestion → preprocessing → training → validation → inference → visualisation
- Applying transfer learning appropriately for a small dataset instead of training from scratch
- Writing PyTorch training loops with correct gradient, loss, and device handling
- Structuring ML code as reusable components (dataset class, model class, `preprocess_image` / `predict` / `visualise` functions)
- Debugging and reasoning about model output via confidence distributions rather than accuracy alone
- Version control with Git/GitHub and documenting a project for other readers

## Project structure

```
.
├── cardPredictor.py          # Training, validation, and interactive prediction
├── requirements.txt          # Python dependencies
└── resources/
    ├── README.md             # Dataset layout, cards.csv schema, and model-file notes
    ├── train/                # Training images, one subfolder per card class
    ├── valid/                # Validation images, one subfolder per card class
    ├── test/                 # Test images, one subfolder per card class
    ├── cards.csv             # Class index, image path, label, card type, split (see resources/README.md)
    ├── 14card types - 14-(200 x 200)-94.61.h5   # Earlier 14-class experiment — see note below
    └── 53cards-53-(200x200)-100.00.h5           # Earlier 53-class experiment — see note below
```

### A note on the `.h5` files

The two `.h5` files are **HDF5-serialised trained model files from earlier experiments** — the filename encodes the experiment: number of classes, input image size (200 × 200), and the accuracy recorded for that run (94.61% for the 14-class version, 100.00% for the 53-class version). They are **not** loaded by the current PyTorch script, which trains an EfficientNet-B0 model in memory; they are kept as artefacts of the earlier modelling work that led to the current approach.

## Getting started

```bash
pip install -r requirements.txt
python cardPredictor.py
```

Dataset paths are resolved relative to the script itself (not your current working directory), so the script can be run from anywhere. When training finishes, it prompts for an image path — enter the path to a card image, or `n` to exit.

## Current status and honest limitations

- **The dataset, `cards.csv`, and the `.h5` model files are not currently committed to this repository** — only the code is. The `resources/train`, `resources/valid`, and `resources/test` folders in GitHub previously contained only placeholder files, so the project cannot be re-run end-to-end from a fresh clone until the data is added. Restoring that data, and saving/loading the trained PyTorch model, are the top next steps.
- Because no dataset or evaluation log is in the repo, **no accuracy figure is claimed for the current PyTorch model.** The percentages in the `.h5` filenames belong to those earlier experiments, not to this script.
- Training runs for a fixed 5 epochs with no model checkpointing, early stopping, data augmentation, or per-class metrics yet.

## Planned improvements

- [ ] Commit (or link) the dataset and `cards.csv`, and document its source and licence
- [ ] Save and load the trained model so inference does not require retraining
- [ ] Report test-set accuracy and a confusion matrix, especially for visually similar cards
- [ ] Add data augmentation and tune epochs / learning rate
- [ ] Add a small CLI or web demo that accepts an uploaded image instead of a local file path
- [ ] Add automated tests for the dataset layout and preprocessing

---

*Built by Shafat (Blindmanzeye) as a hands-on deep learning / computer vision portfolio project.*
