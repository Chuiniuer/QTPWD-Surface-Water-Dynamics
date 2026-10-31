# Lake and river classification

This directory contains scripts for identifying reservoirs, distinguishing lakes from rivers, and assigning lake attributes to monthly surface water objects.

## Workflow

1. `identify_reservoirs.py` identifies reservoir objects using a reservoir inventory.
2. `prepare_references.py` prepares reference masks from river and lake datasets, including preprocessed GRWL and OSM river masks.
3. `label_waterbodies.py` assigns candidate labels, and `split_labels.py` exports objects for manual labelling.
4. `extract_features.py` prepares terrain and distance rasters and extracts features for monthly water objects.
5. `classify_waterbodies.py` trains and evaluates classifiers and predicts lake and river labels.
6. `prepare_lake_type_references.py` prepares glacier and permafrost references, and `classify_lake_types.py` assigns lake attributes.

## Usage

Set the input and output paths in each script before running the workflow. To generate the terrain and distance rasters, set `PREPARE_RASTERS = True` in `extract_features.py`. To train the LightGBM classifier, set `RUN_LIGHTGBM_TEST = True` and `RUN_PREDICTION = False` in `classify_waterbodies.py`. Prediction updates the label field in the source GeoPackages.

The scripts require NumPy, pandas, GDAL, Fiona, GeoPandas, pyogrio, Shapely 2, pyproj, rasterio, PyArrow, scikit-learn, LightGBM, joblib and Matplotlib. Input datasets and labels are not included.
