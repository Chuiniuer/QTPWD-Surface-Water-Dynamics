# Reference dataset

Core scripts for preparing the surface water reference dataset:

- `train_final_rf.py`: train the random forest using labelled point samples.
- `rf_mapping_multi_regions.py`: generate initial water masks.
- `extract_tiles_512_512.py`: extract image patches and initial labels.
- `standardised_configuration.py`: calculate z-score parameters from unnormalised patches in the training, validation and test folders.
- `train_test_split.py`: split and standardise manually corrected patches.

Before running, configure the paths, years and regions in each script. The workflow requires prepared point samples with 17 features and labels, mosaicked initial masks and manually corrected patch labels. The parameter script requires existing unnormalised patches in split folders; the split script requires an existing parameter file.

Requirements: Python 3, GDAL, NumPy, pandas, scikit-learn, joblib and tqdm.
