"""One-hour-ahead rolling forecasts; every feature precedes its target."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

METHODS = ["persistence", "seasonal24", "hist_gradient_boosting"]


def build_features(series: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame(index=series.index)
    for lag in (1, 2, 3, 24, 48, 168):
        result[f"lag_{lag}"] = series.shift(lag)
    for window in (3, 24):
        result[f"mean_{window}"] = series.shift(1).rolling(window).mean()
    result["hour_sin"] = np.sin(2 * np.pi * series.index.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * series.index.hour / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * series.index.dayofweek / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * series.index.dayofweek / 7)
    result["target"] = series
    return result.dropna()


def split_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "train": (index >= "2022-09-01") & (index < "2023-01-01"),
        "validation": (index >= "2023-01-01") & (index < "2023-02-01"),
        "test": (index >= "2023-02-01") & (index < "2023-03-01"),
    }


def run_forecast(ratios: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pieces = []
    fit_times = {}
    with threadpool_limits(limits=1):
        for region in ratios.columns:
            frame = build_features(ratios[region])
            masks = split_masks(frame.index)
            model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0, max_bins=63, random_state=2026)
            features = frame.columns.drop("target")
            start = time.perf_counter()
            model.fit(frame.loc[masks["train"], features], frame.loc[masks["train"], "target"])
            fit_times[region] = time.perf_counter() - start
            for split in ("validation", "test"):
                heldout = frame.loc[masks[split]]
                part = pd.DataFrame({
                    "time": heldout.index, "region": region, "split": split,
                    "actual_ratio": heldout["target"].to_numpy(),
                    "persistence": heldout["lag_1"].to_numpy(),
                    "seasonal24": heldout["lag_24"].to_numpy(),
                    "hist_gradient_boosting": np.clip(model.predict(heldout[features]), 0, 1),
                })
                pieces.append(part)
    predictions = pd.concat(pieces, ignore_index=True)
    metrics = []
    for split in ("validation", "test"):
        frame = predictions[predictions["split"] == split]
        for region in ["all", *ratios.columns]:
            group = frame if region == "all" else frame[frame["region"] == region]
            for method in METHODS:
                error = group[method].to_numpy() - group["actual_ratio"].to_numpy()
                metrics.append({"split": split, "region": region, "method": method, "mae_percentage_points": float(100 * np.abs(error).mean()), "rmse_percentage_points": float(100 * np.sqrt(np.mean(error ** 2))), "n_predictions": len(group)})
    metadata = {
        "seed": 2026, "fit_seconds_by_region": fit_times, "fit_seconds_total": sum(fit_times.values()),
        "model": "HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,learning_rate=0.05,l2_regularization=1,max_bins=63,random_state=2026)",
        "training_rows_per_region": int(split_masks(build_features(ratios.iloc[:, 0]).index)["train"].sum()),
        "validation_hours": 744, "test_hours": 672, "thread_limit": 1,
        "evaluation": "Rolling one-hour-ahead: previously observed held-out hours may feed later held-out forecasts; no refitting or future observations.",
        "validation_use": "Diagnostic only; model parameters fixed before examining validation and test scores.",
    }
    return predictions, pd.DataFrame(metrics), metadata
