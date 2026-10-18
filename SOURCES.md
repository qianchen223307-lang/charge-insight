# Sources, ownership and data interpretation

The forecasting and reporting code in this repository was independently implemented. UrbanEV observations are third-party research data, not data collected by this account.

## Data

- Authors' repository: https://github.com/IntelligentSystemsLab/UrbanEV
- Fixed source commit: `44f2aa0c8d89f192bce00bafb0def74a21b39c68`.
- `occupancy.csv`: Git blob `d1f03f65919fa1bf2872c5bb8b500cb90e7d71fb`; SHA-256 `1b9099b8c6c33510a2c48a79c4b524b01f6a2f5df45b9d604361b22ad4bc5211`.
- `inf.csv`: Git blob `4b9efa0f831f7e7c039c907f86d9a52f9aa0a686`; SHA-256 `03c9830965e9e99b29adfb8cceed0eba98d37631f514273cb3fe61f80d63de7c`.
- License: [CC0 1.0](https://github.com/IntelligentSystemsLab/UrbanEV/blob/44f2aa0c8d89f192bce00bafb0def74a21b39c68/LICENSE). Code license here is MIT; source data retain their separate CC0 dedication.
- Citation: Li, H., Qu, H., Tan, X. et al. *UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction*. Scientific Data 12, 523 (2025). https://doi.org/10.1038/s41597-025-04874-4

Exact source bytes were retrieved through GitHub's blob API, verified against both fixed Git blob IDs, and reduced to the first 16 region columns. Only the 264 KB subset and aggregated capacities are bundled. `data/provenance.json` records exact hashes and the deterministic selection rule. `python -m charge_insight download` can independently fetch the same commit over HTTPS, verify its Git blob hashes and regenerate the bundle.

## Correcting the normalization

The upstream README calls `occupancy.csv` a percentage. The actual source has 4,344 hourly rows and 275 region columns, numeric values from 0 to 373. The authors' [`read_data`](https://github.com/IntelligentSystemsLab/UrbanEV/blob/44f2aa0c8d89f192bce00bafb0def74a21b39c68/code/utils.py) divides these values by charging-pile counts. We therefore interpret the raw values as occupied/unavailable pile counts, and explicitly publish a derived ratio.

`inf.csv` has 1,362 station records covering 275 regions, with 17,532 piles in total. A region can contain multiple stations: **sum `charge_count` grouped by `TAZID`**. The upstream `dict(zip(TAZID, charge_count))` expression keeps only the last station per region, so this implementation does not reuse it. Across all source regions, raw observations do not exceed summed regional capacity; the maximum derived ratio is 1.

The fixed first 16 regions are `102,104,105,106,107,108,109,110,111,115,123,124,202,204,205,206`. Selection precedes model evaluation and is unrelated to test performance.

## Limits and independent contribution

The independent contribution is source integrity checking, station-aware aggregation, a leakage-conscious feature/evaluation pipeline, executable CPU baselines, business tests and auditable outputs. It is not a reproduction of the authors' graph/Transformer models or a claim of a new forecasting method.

The source has already undergone author-side outlier replacement and forward/backward filling. The original missingness mask is unavailable. This benchmark cannot establish performance on unprocessed streaming data. Time strings are treated as the source's local Shenzhen calendar; no conversion to a different timezone is applied.

`volume.csv` is deliberately excluded: the authors describe its energy as rated pile power multiplied by duration. Neither it nor the occupancy observations are measured electrical power curves. No carbon-saving or operational-deployment claim is made.

Model implementation documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
