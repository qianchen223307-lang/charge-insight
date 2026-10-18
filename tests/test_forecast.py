import unittest

import numpy as np
import pandas as pd

from charge_insight.data import capacities_from_stations, git_blob_hash, load_data
from charge_insight.forecast import build_features, split_masks


class ForecastTests(unittest.TestCase):
    def test_station_capacities_are_summed(self):
        stations = pd.DataFrame({"TAZID": [102, 102, 104], "charge_count": [10, 20, 5]})
        self.assertEqual(capacities_from_stations(stations)["102"], 30)

    def test_features_do_not_read_target_or_future(self):
        index = pd.date_range("2022-09-01", periods=300, freq="h")
        a = pd.Series(np.arange(300, dtype=float), index=index)
        b = a.copy()
        b.iloc[250:] = -999
        fa, fb = build_features(a), build_features(b)
        pd.testing.assert_series_equal(fa.loc[index[250]].drop("target"), fb.loc[index[250]].drop("target"))
        self.assertEqual(fa.loc[index[250], "lag_1"], 249)
        self.assertEqual(fa.loc[index[250], "mean_3"], 248)

    def test_target_time_controls_split(self):
        index = pd.DatetimeIndex(["2022-12-31 23:00", "2023-01-01", "2023-02-01"])
        masks = split_masks(index)
        self.assertEqual(masks["train"].tolist(), [True, False, False])
        self.assertEqual(masks["validation"].tolist(), [False, True, False])
        self.assertEqual(masks["test"].tolist(), [False, False, True])

    def test_bundled_source_integrity_and_normalization(self):
        ratios, capacity, provenance = load_data()
        self.assertEqual(ratios.shape, (4344, 16))
        self.assertEqual(int(capacity["102"]), 30)
        self.assertAlmostEqual(ratios.iloc[0, 0], 17 / 30)
        self.assertTrue(((ratios >= 0) & (ratios <= 1)).all().all())
        self.assertEqual(len(provenance["sources"][0]["sha256"]), 64)

    def test_git_blob_hash(self):
        self.assertEqual(git_blob_hash(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")


if __name__ == "__main__":
    unittest.main()
