# Water dynamics

Core scripts for analysing the monthly QTPWD water masks. Set the paths and processing years in each script before running; the supplied paths reflect the original working environment.

## Persistence and transitions

- `annual_persistence.py` calculates annual inundation frequency, temporal heterogeneity and persistence classes from monthly masks encoded as 0, 1 and 2. Its strict annual classification can leave valid pixels unclassified (254).
- `persistence_transitions.py` calculates five-year mean F and H and class transitions. It uses a complete persistence classification whose conditions differ from the annual script. Point its input directory to the annual F/H outputs.

## Waterbody area and seasonality

- `vectorise_water.py` converts water masks to waterbody polygons. This script requires ArcGIS Pro with ArcPy and Spatial Analyst; its example configuration covers 2023–2025.
- `waterbody_centroids.py` creates centroid points from waterbody polygons for the size statistics below. Set its input directory to the vectorisation output; this step can be skipped if the points are already available.
- `waterbody_statistics.py` produces monthly plateau and watershed statistics by waterbody size. Watersheds are assigned by centroid, and each assigned waterbody contributes its full area. Waterbodies below 0.1 km² contribute to small-water area but are excluded from counts.
- `aggregate_area_25km.py` aggregates water masks to 25 km grid cells, producing water area and valid-observation proportion. It does not separate waterbody size classes.
- `seasonality.py` calculates plateau and watershed seasonality dominance and decomposes monthly area series for size-class contributions to total seasonal variation. It requires the two monthly statistics CSV files and their companion Excel workbook.
- `seasonality_grid.py` reprojects monthly grid rasters and calculates 25 km seasonality dominance and seasonal change, producing the grid-centre layer used by Figure 4. Check the configured input grid, output grid and data-validity thresholds before running.

## Extremes

- `extremes.py` calculates regional and grid-level low- and high-water indicators using the monthly statistics and 25 km rasters. The significance tests and impact categories used in Figure 4 are implemented in the figure script.

Each script can be run with `python water_dynamics/<script>.py`. The seasonality and extremes scripts also support command-line paths; use `--help` for options. Several scripts use different default data directories, so ensure outputs and subsequent inputs point to the same files.

Requirements include NumPy, pandas, Rasterio, GDAL, Numba, SciPy, statsmodels, GeoPandas, Pyogrio, pyshp, openpyxl and tqdm. ArcGIS Pro is required only for `vectorise_water.py`.
