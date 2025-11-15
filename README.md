# House Price Prediction — EDA & Modeling

## Project Overview

This repository contains a complete end-to-end analysis and model pipeline for predicting house prices using a real estate listings dataset. The focus is on thorough exploratory data analysis (EDA), careful data cleaning and outlier handling, feature engineering, and training/evaluating regression models (linear regression, Lasso, decision tree). The goal is a clear, reproducible pipeline that turns raw messy listings into a reliable price-prediction model.

---

## Highlights

* Data cleaning and preprocessing tailored for real-estate: handling inconsistent `total_sqft`, `price_per_sqft` calculations, missing values, and categorical encoding of `location`.
* EDA visualizations (scatter plots, histograms, boxplots) to reveal the relationships between `total_sqft`, `bhk`, `bath`, and `price`.
* Outlier removal heuristics implemented per-location (price-per-sqft based) and per-`bhk` to reduce noise and mislabeled data.
* Modeling using multiple algorithms with cross-validation and grid search to find the best performing model.
* Utility function for single-point predictions (`predict_price`) that maps user input to model features.

---

## Files in this Project

* `project(house-price-prediction).ipynb` — Jupyter notebook with the full EDA, preprocessing pipeline, model training, and evaluation steps.
* `data/` — folder (recommended) where raw/cleaned CSVs are stored (not included if you have private data).
* `README.md` — this file.

---

## Dataset

The dataset should be a table of house/listing records with, at minimum, the following columns:

* `location` — string (city/neighborhood)
* `total_sqft` — numeric (total area in square feet; can be cleaned from ranges)
* `price` — numeric (listing price, in lakhs or consistent unit)
* `bhk` — integer (number of bedrooms)
* `bath` — integer (number of bathrooms)

Additional derived column used in the notebook:

* `price_per_sqft` = `price * 1e5 / total_sqft` (or kept in lakhs-per-sqft depending on your scale)

> Note: The notebook includes helper code to normalize inconsistent `total_sqft` entries (e.g., ranges like "2100 - 2850") and to convert price units where necessary.

---

## Main Steps in the Notebook

1. **Load data** — Read CSV into `pandas.DataFrame` and inspect basic info/shape.
2. **Data cleaning**

   * Remove rows with missing or invalid `total_sqft` and `price`.
   * Parse `total_sqft` ranges into numeric averages where needed.
   * Convert textual/scale differences in `price` to a consistent unit.
   * Create `price_per_sqft` column for outlier detection.
3. **Exploratory Data Analysis (EDA)**

   * Summary statistics (`describe()`), distribution plots, and correlation checks.
   * Scatter plots of `price` vs `total_sqft` split by `bhk` to visually inspect anomalies.
   * Boxplots by `location` or `bhk` to find skew and dispersion.
4. **Outlier Removal**

   * `remove_pps_outliers`: per-location filter using price-per-sqft mean ± k*std (or z-score). This reduces extreme per-sqft values.
   * `remove_bhk_outliers`: heuristic that removes listings where a higher-`bhk` has price_per_sqft lower than the average of the lower-`bhk` (e.g., a 3-BHK cheaper per-sqft than average 2-BHK in the same location).
5. **Feature Engineering**

   * One-hot encode `location` (with optional `drop_first` or grouping rare locations as `other`).
   * Ensure numeric columns are in the correct order and scale.
6. **Model Training and Evaluation**

   * Split data into `X` (features) and `y` (target), then into train/test sets.
   * Train `LinearRegression`, evaluate with `r2`, `MAE`, and `RMSE`.
   * Use `ShuffleSplit` cross-validation and `cross_val_score` to get a stable estimate.
   * Use `GridSearchCV` with pipelines (recommended) to tune `Lasso` and `DecisionTreeRegressor` hyperparameters.
7. **Prediction Utility**

   * `predict_price(location, sqft, bath, bhk)` function to construct a feature vector and produce a prediction from the trained model.

---

## How to run

1. Create a Python environment (recommended tools: `venv` or `conda`).
2. Install dependencies:

```bash
pip install -r requirements.txt
# or, if requirements.txt isn't provided:
pip install pandas numpy matplotlib scikit-learn jupyter
```

3. Place your dataset CSV in the `data/` directory and update the notebook path if required.
4. Open and run the Jupyter notebook:

```bash
jupyter notebook project\(house-price-prediction\).ipynb
```

5. Step through the notebook cells to reproduce the EDA, cleaning, and model training.

---

## Dependencies

* Python 3.8+
* pandas
* numpy
* matplotlib
* scikit-learn
* jupyter (optional, for interactive use)

You can pin exact versions by creating a `requirements.txt` from your environment.

---

## Results & Evaluation

* The notebook prints model performance metrics (train/test R², cross-validated scores) and displays diagnostic plots (predicted vs actual, residuals).
* The README intentionally omits numbers because they vary by the dataset and the exact pipeline choices (outlier thresholds, encoding).

---
