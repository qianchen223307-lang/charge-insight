# Charge Insight

**Auditable one-hour-ahead regional EV charging occupancy forecasts using real, author-preprocessed UrbanEV observations.**

[Live presentation](https://qianchen223307-lang.github.io/charge-insight/) · [中文](README.md) · [Sources and interpretation](SOURCES.md)

Independent work here includes pinned-data verification, station-aware capacity aggregation, strictly past-facing features, CPU baselines, tests and exported predictions. This is not a reproduction of UrbanEV's published graph or Transformer models.

## Reproduce

With Python 3.12, install once and run from the repository root:

```bash
python -m pip install -e .
python -m charge_insight demo
python -m unittest discover -s tests -v
```

The demo runs offline on the included 264 KB subset and rewrites the actual predictions, per-region/aggregate MAE and RMSE, environment and measured fit times under `results/`, plus presentation data under `docs/data.json`. To reacquire the same source files and verify their fixed Git blob hashes, run `python -m charge_insight download`.

## Protocol

The first 16 official region columns are fixed before evaluation. Raw occupied-pile counts are divided by the **sum of all station capacities in each region**. The source README's percentage description differs from the CSV values; the author's normalization code and all-region capacity bounds support this explicit correction. Duplicate region IDs must be aggregated rather than overwritten.

Training targets span September–December 2022 (2,760 per region after 168 history hours); January 2023 is diagnostic validation (744 hours), and February is the fixed test (672 hours; 10,752 region-hours). Inputs use past lags, shifted rolling means and known calendar fields. Fixed-parameter HistGradientBoosting is compared with persistence and previous-day baselines. No hyperparameter is selected using the test set. Evaluation is rolling one step ahead, not a month-long forecast made at one origin.

## Actual run

| Method | February test MAE (percentage points) | RMSE (percentage points) |
|---|---:|---:|
| Persistence | 1.0682 | 2.3570 |
| Previous-day seasonal | 3.6312 | 5.5084 |
| HistGradientBoosting | 1.6461 | 2.6402 |

**Persistence outperforms the boosting model on this fixed holdout.** This negative result is retained, not reframed as a model improvement. Measured training time and the environment are recorded in `results/run.json`; they vary with the machine.

## Data and ownership boundaries

UrbanEV has already undergone author-side outlier correction and forward/backward filling; the original missingness mask is unavailable. This cannot establish unprocessed streaming performance, cross-city generalization, actual energy savings or deployment. Occupancy is not measured electrical load. The estimated-energy `volume.csv` is not used.

Code: MIT. Source observations: UrbanEV, CC0; see the fixed source commit, hashes and paper citation in [SOURCES.md](SOURCES.md). Code and documentation were developed with AI assistance; see [AI_USE.md](AI_USE.md).
