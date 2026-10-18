"""Pinned source acquisition and validated, station-aware normalization."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_COMMIT = "44f2aa0c8d89f192bce00bafb0def74a21b39c68"
SOURCE_BASE = f"https://raw.githubusercontent.com/IntelligentSystemsLab/UrbanEV/{SOURCE_COMMIT}"
SOURCE_BLOBS = {
    "occupancy.csv": "d1f03f65919fa1bf2872c5bb8b500cb90e7d71fb",
    "inf.csv": "4b9efa0f831f7e7c039c907f86d9a52f9aa0a686",
}
REGIONS = ["102", "104", "105", "106", "107", "108", "109", "110", "111", "115", "123", "124", "202", "204", "205", "206"]
ROOT = Path(__file__).resolve().parents[1]


def git_blob_hash(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def capacities_from_stations(stations: pd.DataFrame) -> pd.Series:
    """Sum all stations in each traffic zone; duplicate zones must not overwrite."""
    required = {"TAZID", "charge_count"}
    if not required.issubset(stations.columns):
        raise ValueError(f"station data requires {sorted(required)}")
    work = stations.copy()
    work["TAZID"] = work["TAZID"].astype(str)
    work["charge_count"] = pd.to_numeric(work["charge_count"], errors="raise")
    if (work["charge_count"] <= 0).any():
        raise ValueError("station capacities must be positive")
    return work.groupby("TAZID")["charge_count"].sum()


def prepare_source(occupancy_bytes: bytes, station_bytes: bytes, data_dir: Path) -> dict:
    raw = {"occupancy.csv": occupancy_bytes, "inf.csv": station_bytes}
    for name, content in raw.items():
        if git_blob_hash(content) != SOURCE_BLOBS[name]:
            raise ValueError(f"source integrity check failed: {name}")
    occupancy = pd.read_csv(io.BytesIO(occupancy_bytes))
    if list(occupancy.columns[1:17]) != REGIONS:
        raise ValueError("the pinned source column order has changed")
    stations = pd.read_csv(io.BytesIO(station_bytes))
    capacity = capacities_from_stations(stations).loc[REGIONS]
    subset = occupancy[["time", *REGIONS]]
    data_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(data_dir / "urbanev_first16.csv", index=False, float_format="%.6g", lineterminator="\n")
    capacity.rename("capacity").rename_axis("region").to_csv(data_dir / "capacities.csv", lineterminator="\n")
    provenance = {
        "dataset": "UrbanEV", "repository": "https://github.com/IntelligentSystemsLab/UrbanEV",
        "commit": SOURCE_COMMIT, "license": "CC0-1.0", "regions": REGIONS,
        "selection": "First 16 columns of the pinned official occupancy.csv; independent of forecast scores",
        "sources": [{"path": name, "url": f"{SOURCE_BASE}/data/{name}", "git_blob_sha1": SOURCE_BLOBS[name], "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)} for name, content in raw.items()],
        "files": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in ("urbanev_first16.csv", "capacities.csv")},
        "normalization": "sum station charge_count by TAZID, then divide each region's raw occupied-pile count by its capacity",
        "source_preprocessing": "Author-cleaned data includes outlier replacement and forward/backward filling; no original missingness mask is supplied.",
    }
    (data_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return provenance


def download(data_dir: Path | None = None) -> dict:
    payload = {}
    for name in SOURCE_BLOBS:
        request = urllib.request.Request(f"{SOURCE_BASE}/data/{name}", headers={"User-Agent": "charge-insight/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload[name] = response.read()
    return prepare_source(payload["occupancy.csv"], payload["inf.csv"], data_dir or ROOT / "data")


def load_data(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.Series, dict]:
    data_dir = data_dir or ROOT / "data"
    provenance = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    for name, digest in provenance["files"].items():
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"bundled data integrity check failed: {name}")
    raw = pd.read_csv(data_dir / "urbanev_first16.csv", parse_dates=["time"]).set_index("time")
    caps = pd.read_csv(data_dir / "capacities.csv", dtype={"region": str}).set_index("region")["capacity"]
    if list(raw.columns) != REGIONS or len(raw) != 4344:
        raise ValueError("expected the pinned 4344-hour, 16-region subset")
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("timestamps must be unique and increasing")
    if not (raw.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("hourly coverage must be complete")
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("occupancy must be finite")
    ratio = raw.div(caps, axis=1)
    if ((ratio < 0) | (ratio > 1)).any().any():
        raise ValueError("occupancy is outside regional capacity")
    return ratio, caps, provenance
