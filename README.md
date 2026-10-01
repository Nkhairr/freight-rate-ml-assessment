# Freight Rate Prediction Challenge

## Approach

The solution uses a chronological validation design: January–August 2025 for training and September–October 2025 for validation. This better represents the production task, where the final predictions are for later dates (November–December), and avoids leakage from future observations.

The model is a `CatBoostRegressor` trained with MAE loss. CatBoost was selected because it handles nonlinear relationships, missing numeric values, and categorical pickup, delivery, equipment, and route variables without one-hot dimensionality expansion. Date-derived features include month, day of week, day of month, day of year, ISO week, and cyclic annual encodings. A combined pickup/delivery route feature is also included.

Data quality handling:

- Missing `weight` and `market_index` values are retained as missing, with explicit missingness indicators.
- Non-positive weights are treated as invalid and set to missing; the invalidity flag is retained.
- The extreme target values are not removed from final training; MAE loss reduces their influence on the central prediction.
- December inputs do not contain the optional coordinate and market columns, so the inference code aligns them to the training schema and supplies missing values safely.

## Run

```bash
python -m pip install -r requirements.txt
python solution.py
python score.py \
  --predictions validation_predictions.csv \
  --december-predictions december-chart-inputs.csv
```

The command creates `validation_predictions.csv`, completes `december-chart-inputs.csv`, and generates `scorer_results/candidate_december.png`.

## Files

- `solution.py`: feature engineering, training, and inference.
- `validation_predictions.csv`: 12,000 final validation predictions.
- `december-chart-inputs.csv`: completed 31-day December prediction input.
- `scorer_results/candidate_december.png`: scorer-generated fixed December chart.
- `report.pdf`: methodology and results report.
