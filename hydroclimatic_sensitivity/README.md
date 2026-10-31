# Hydroclimatic sensitivity

These scripts assess the sensitivity of surface water area and seasonality to nine hydroclimatic variables in 25 km grids.

- `multicollinearity.py` calculates pooled tolerance and variance inflation factors and prepares the model dataset.
- `validation.py` evaluates grid-specific random forest models.
- `sensitivity.py` estimates responses to a one-standard-deviation increase in each variable.

Set the input and output paths in each script. Run `validation.py` before `sensitivity.py`. The scripts require prepared grid-level monthly water and hydroclimatic data and associated spatial references; these inputs are not included. The initial eligibility screening used fields beyond the final nine variables. Validation of seasonal targets uses seasonal components estimated from the complete monthly record.

Dependencies include NumPy, pandas, scikit-learn, statsmodels, joblib, Fiona, Shapely, pyproj and Matplotlib.
