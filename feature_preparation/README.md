# Feature preparation

These scripts calculate water indices, principal components and GLCM texture features from Sentinel-2 imagery.

Keep `compute_features.py` and `fast_glcm.py` in the same folder. The main script imports the GLCM functions automatically.

Requirements: Python 3, GDAL, NumPy, pandas, OpenCV and tqdm.

Set `REGION_ID`, `START_YEAR`, `END_YEAR`, `INPUT_FOLDER` and `OUTPUT_ROOT` in `compute_features.py`. The input folder should contain B2, B3, B4, B8, B11 and B12 files named `{region}_sentinel2_{YYYY}{MM}_{band}.tif`.

Run:

```bash
python compute_features.py
```
