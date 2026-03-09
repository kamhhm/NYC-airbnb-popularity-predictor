# Predicting NYC Airbnb Listing Popularity

An end-to-end scikit-learn regression pipeline that predicts Airbnb listing popularity (`reviews_per_month`) using only features available before a listing goes live:

- **Numeric:** `price`, `latitude`, `longitude`, `minimum_nights`, `calculated_host_listings_count`, `availability_365`
- **Categorical:** `neighbourhood_group` (borough), `neighbourhood` (220 unique), `room_type`
- **Text:** `name` (listing title, vectorized via TF-IDF)

## Approach

- Mixed-type `ColumnTransformer` preprocessing pipeline: `StandardScaler` for numerics, `OneHotEncoder` for 220+ neighbourhoods, and `TfidfVectorizer` for listing-title NLP
- Data leakage prevention by identifying and excluding review-derived features
- Log-target regression via `TransformedTargetRegressor` to handle heavy right-skew
- Systematic model comparison (Ridge, Random Forest, HistGradientBoosting, Stacking) with cross-validation
- Hyperparameter tuning with `GridSearchCV` and permutation importance analysis

## Tech Stack

Python, scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

## Dataset

[New York City Airbnb Open Data](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) — ~48K listings, 2019. Download `AB_NYC_2019.csv` and place it in `data/`.

## Setup

```bash
pip install -r requirements.txt
jupyter lab notebook.ipynb
```
