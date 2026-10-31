# Monthly surface water gap filling

These scripts fill missing observations in monthly water masks using historical water frequency and spatial and temporal neighbourhood similarity.

## Input

Provide a complete sequence of aligned monthly rasters named `QTP_water_mask_YYYY_MM.tif` or `QTP_water_mask_YYYY_MM_clip.tif`, with values 0 for non-water, 1 for water and 2 for invalid observations. A fixed plateau mask aligned to the monthly rasters is also required, with non-zero values inside the plateau and zero outside.

## Run

Set the input, output and fixed-mask paths in both scripts. From the repository root:

```bash
python gap_filling/gap_stats.py
python gap_filling/stns_gap_fill.py
```

The first script generates `gap_counts_block_month.npz`, which the filling script uses to distribute work. Set `GAP_STAT_ROOT` in `stns_gap_fill.py` to the statistics output directory. Both scripts must use the same year range and processing block size.

Outputs are written as `YYYY/monthMM/QTP_water_mask_STGF_YYYY_MM.tif` under the filling output directory. Existing valid observations are retained. Gaps without valid historical observations remain invalid; equal water and non-water scores are assigned to non-water.

The implementation uses fixed neighbourhood-size denominators for similarity and adds a small constant to the spatial and temporal scores. Time-similarity reference years exclude the first and last years of the input series. See the scripts for the full configuration.

Requirements: Python 3, NumPy, OpenCV, GDAL and tqdm.
