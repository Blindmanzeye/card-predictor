# resources/ — dataset and model files

This folder holds everything `cardPredictor.py` trains and predicts from.
Paths in the script are resolved relative to the project root (the folder
containing `cardPredictor.py`), not relative to your current working
directory, so the layout below must be matched exactly.

## Expected layout

```
resources/
├── train/                 # training split
│   └── <card class>/      # one folder per card, e.g. "ace of spades"
│       └── *.png          # 1–5 images of that card
├── valid/                 # validation split, same per-class structure
│   └── <card class>/
│       └── *.png
├── test/                  # test split, same per-class structure
│   └── <card class>/
│       └── *.png
├── cards.csv
├── 14card types - 14-(200 x 200)-94.61.h5
└── 53cards-53-(200x200)-100.00.h5
```

Rules that matter (the training code relies on `torchvision.datasets.ImageFolder`):

- Images must be **inside a subfolder named after their card class**.
  An image file placed directly in `train/`, `valid/`, or `test/` — for
  example the placeholder `img.png` files that were previously in those
  folders — is not a class, and it breaks dataset loading.
- Every split should contain the **same set of class folders**.
- Empty splits are kept in Git with a `.gitkeep` file, because Git does
  not track empty folders.

## cards.csv

One row per image, with these columns:

| Column | Meaning |
|---|---|
| class index | Integer ID of the card class |
| file path | Path to the image, relative to `resources/` |
| class label | Label for the class |
| card type | The playing card the image shows |
| dataset indicator | Which split the image belongs to (`train`, `valid`, or `test`) |

`cards.csv` is documentation/index data for the dataset. The current
PyTorch script does not read it — it discovers classes from the folder
structure above — so the two must be kept in agreement by hand.

## The .h5 model files

`.h5` is the **HDF5** format commonly used to save a trained neural
network (weights, and sometimes the full model) — most often by
Keras/TensorFlow. The filenames here encode the experiment that produced
each file:

| File | Classes | Input size | Figure in the name |
|---|---|---|---|
| `14card types - 14-(200 x 200)-94.61.h5` | 14 card types | 200 × 200 | 94.61 |
| `53cards-53-(200x200)-100.00.h5` | 53 cards (52 + joker) | 200 × 200 | 100.00 |

The trailing figure is the accuracy recorded for that run when the file
was named. These are artefacts of earlier experiments; the current
`cardPredictor.py` is a PyTorch/EfficientNet model that trains in memory,
resizes images to 128 × 128, and neither saves nor loads `.h5` files.
Treat the `.h5` accuracies as historical filenames, not as a measured
result for the current script.

## Current status

As of the structure-fix PR, the dataset images, `cards.csv`, and the
`.h5` files above are **not in this GitHub repository** — the splits
contain only `.gitkeep` placeholders. Add them here, in the layout above,
from the local copy of the project to make the repository runnable
end-to-end. Watch GitHub's file-size limits before committing large
`.h5` files or the full image set; Git LFS may be the better route.
