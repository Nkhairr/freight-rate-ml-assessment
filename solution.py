from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pandas.api.types import is_numeric_dtype

ROOT = Path(__file__).resolve().parent


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    if "posted_rate" in d:
        d = d.drop(columns=["posted_rate"])
    if "load_id" in d:
        d = d.drop(columns=["load_id"])
    dates = pd.to_datetime(d.pop("date"))
    for column in [
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "market_index", "quote_signal", "weight",
    ]:
        if column not in d:
            d[column] = np.nan
    d["month"] = dates.dt.month.astype(str)
    d["dayofweek"] = dates.dt.dayofweek.astype(str)
    d["dayofmonth"] = dates.dt.day.astype(np.int16)
    d["dayofyear"] = dates.dt.dayofyear.astype(np.int16)
    d["weekofyear"] = dates.dt.isocalendar().week.astype(np.int16)
    d["year_progress_sin"] = np.sin(2 * np.pi * dates.dt.dayofyear / 365.25)
    d["year_progress_cos"] = np.cos(2 * np.pi * dates.dt.dayofyear / 365.25)
    d["weight_missing"] = d["weight"].isna().astype(str)
    d["weight_invalid"] = (d["weight"].fillna(0) <= 0).astype(str)
    d.loc[d["weight"] <= 0, "weight"] = np.nan
    d["market_missing"] = d["market_index"].isna().astype(str)
    d["route"] = d["pickup"].astype(str) + "__" + d["delivery"].astype(str)
    d["distance_log"] = np.log1p(d["distance"])
    return d


def main() -> None:
    train = pd.read_csv(ROOT / "train-test.csv")
    validation = pd.read_csv(ROOT / "validation.csv")
    december = pd.read_csv(ROOT / "december-chart-inputs.csv")

    x_train = make_features(train)
    x_validation = make_features(validation)
    x_december = make_features(december)
    x_validation = x_validation.reindex(columns=x_train.columns)
    x_december = x_december.reindex(columns=x_train.columns)
    categorical = [c for c in x_train.columns if not is_numeric_dtype(x_train[c])]
    for frame in (x_train, x_validation, x_december):
        frame[categorical] = frame[categorical].fillna("MISSING")

    # MAE is robust to the small number of extreme posted_rate outliers.
    model = CatBoostRegressor(
        iterations=360,
        depth=7,
        learning_rate=0.04,
        loss_function="MAE",
        l2_leaf_reg=8,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(x_train, train["posted_rate"], cat_features=categorical)

    validation_prediction = np.maximum(model.predict(x_validation), 0.01)
    template = pd.read_csv(ROOT / "validation-predictions-template.csv")
    output = template.copy()
    output["predicted_rate"] = validation_prediction
    output.to_csv(ROOT / "validation_predictions.csv", index=False)

    december_output = december.copy()
    december_output["predicted_rate"] = np.maximum(model.predict(x_december), 0.01)
    december_output.to_csv(ROOT / "december-chart-inputs.csv", index=False)
    print("Created validation_predictions.csv and completed december-chart-inputs.csv")


if __name__ == "__main__":
    main()
