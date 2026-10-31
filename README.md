# FGSA U-Net for surface water mapping

FGSA U-Net is a semantic segmentation model for monthly surface water mapping on the Qinghai–Tibet Plateau. It combines 17 spectral, water index, terrain and texture channels through feature grouping and spatial attention in a U-Net architecture.

The model definition is in [`FGSA_UNet/model.py`](FGSA_UNet/model.py). The training script is based on the full-dataset training workflow, including checkpoint recovery.

## Requirements

Python 3, PyTorch, NumPy and tqdm are required for training. The supplied training script requires a CUDA-enabled GPU. Raster inference also requires GDAL.

## Training data

Arrange the prepared patches as follows, using matching `.npy` filenames for each image, label and valid mask:

```text
patches/
├── train/
│   ├── images_norm/
│   ├── labels_npy/
│   └── valid_masks/
├── val/
│   ├── images_norm/
│   ├── labels_npy/
│   └── valid_masks/
└── test/
    ├── images_norm/
    ├── labels_npy/
    └── valid_masks/
```

Images contain 17 standardised channels with shape `(17, height, width)`; labels and masks contain one value per pixel. The channel order is documented in `model.py`. Standardisation parameters and prepared patches are not included.

## Train the model

From the repository root:

```bash
python FGSA_UNet/train.py --data-root "path/to/patches" --output-dir "path/to/model_output"
```

The default configuration uses 200 epochs, a batch size of four and the AdamW optimiser. `--epochs`, `--batch-size` and `--num-workers` can be set on the command line. The best validation checkpoint is saved as `best_model.pth`.

By default, training resumes from `last_model.pth` in the output directory, or from `checkpoint_epoch_040.pth` if the former is absent. If neither exists, training starts at epoch 1. Add `--no-resume` to start a new run. The saved last checkpoint can contain the previous best validation score when its own epoch sets a new best; retain `best_model.pth` and inspect the training log when resuming.

## Use the model

```python
import torch
from FGSA_UNet.model import FGSAUNet

model = FGSAUNet(in_channels=17, num_classes=1)
images = torch.randn(1, 17, 512, 512)
logits = model(images)
probabilities = torch.sigmoid(logits)
```

For monthly raster inference and assembly of plateau-wide masks, see [`surface_water_mapping/`](surface_water_mapping/README.md). Trained weights and input data are not included.
