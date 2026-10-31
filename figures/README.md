# Main figure scripts

This folder contains the Python scripts used to generate the six main figures for the Qinghai–Tibet Plateau surface water study.

| Script | Figure |
| --- | --- |
| `Figure1.py` | Surface water area and dynamics |
| `Figure2.py` | Surface water persistence |
| `Figure3.py` | Water persistence transitions |
| `Figure4.py` | Seasonality and water extremes |
| `Figure5.py` | Lake types and seasonal dynamics |
| `Figure6.py` | Hydroclimatic sensitivity of surface water seasonality |

## Requirements

Python 3 with NumPy, pandas, Matplotlib, SciPy, statsmodels, Rasterio, GeoPandas, Fiona, Shapely 2 or later, and PyProj. The figures use Times New Roman when available.

## Running the scripts

The scripts use the original project input paths and require the corresponding data and, where applicable, existing analysis results or caches. Check the paths at the beginning of each script before running. The data are not included in this folder.

Run each script separately, for example:

```bash
python Figure1.py
```

Figures are saved by default to:

```text
D:\phdl2\QTP surface water dynamics\code availability\test_output
```

The output directory can be changed in each script. Figures 2, 3 and 5 also use separate directories for intermediate files or caches.
