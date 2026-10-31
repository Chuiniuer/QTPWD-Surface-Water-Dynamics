# Surface water mapping

`inference.py` applies the trained model to monthly feature rasters. `merge_qtp_masks.py` mosaics regional masks and clips them to the plateau grid.

Set input and output paths, region and month selections, the model checkpoint and z-score parameters in each script. The 17 input rasters must share a grid. From the repository root, run:

```bash
python surface_water_mapping/inference.py
python surface_water_mapping/merge_qtp_masks.py
```

The inference script loads the model from `FGSA_UNet/model.py` and saves masks with water, non-water and invalid pixels. The supplied configuration uses 512-pixel windows and a 480-pixel stride, as in the source script. It applies z-score standardisation without clipping; the reference dataset split script clips standardised values to ±5 by default. Check the settings used in the final product before reproducing it. The mosaic step can continue when regions are missing, so check coverage before merging.
