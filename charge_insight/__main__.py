from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .data import ROOT, download, load_data
from .forecast import METHODS, run_forecast


def demo(output_dir: Path | None = None) -> dict:
    # Restrict this CLI process; also avoids unavailable Windows core-discovery utilities.
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    ratios, capacities, provenance = load_data()
    predictions, metrics, metadata = run_forecast(ratios)
    output = output_dir or ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.8f")
    metrics.to_csv(output / "metrics.csv", index=False, float_format="%.8f")
    metadata.update({"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "source_commit": provenance["commit"]})
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    overall = metrics[(metrics["region"] == "all") & (metrics["split"] == "test")].set_index("method")
    model_mae = float(overall.loc["hist_gradient_boosting", "mae_percentage_points"])
    reference_mae = float(overall.loc["persistence", "mae_percentage_points"])
    daily = predictions[predictions["split"] == "test"].copy()
    daily["day"] = pd.to_datetime(daily["time"]).dt.strftime("%m-%d")
    daily = daily.groupby("day")[["actual_ratio", *METHODS]].mean() * 100
    names = {"persistence": "上一小时 / Persistence", "seasonal24": "昨日同小时 / Seasonal 24h", "hist_gradient_boosting": "HistGradientBoosting"}
    payload = {
        "slug": "charge-insight", "title": "Charge Insight", "tagline": "真实区域占用数据 × 可复跑的小时预测", "category": "Energy & forecasting", "data_kind": "real_observations_preprocessed",
        "summary": "独立实现 UrbanEV 数据口径修正、时间留出评估和 CPU 预测基线。预测区域充电桩占用比例，不把估计电量当实测功率。",
        "metrics": [
            {"label": "模型测试 MAE", "value": round(model_mae, 3), "unit": "百分点"},
            {"label": "持续性基线 MAE", "value": round(reference_mae, 3), "unit": "百分点"},
            {"label": "测试覆盖", "value": 16 * 672, "unit": "区域小时"},
            {"label": "实测训练耗时", "value": round(metadata["fit_seconds_total"], 3), "unit": "s", "detail": "16 个区域模型，当前机器单线程；不代表能耗"},
        ],
        "charts": [
            {"title": "2023 年 2 月：每日平均占用比例", "kind": "line", "labels": daily.index.tolist(), "series": [{"name": "实际观测" if col == "actual_ratio" else names[col], "values": daily[col].round(4).tolist()} for col in ["actual_ratio", *METHODS]], "y_label": "占用比例 (%)"},
            {"title": "时间留出测试误差", "kind": "bar", "labels": [names[m] for m in METHODS], "series": [{"name": "MAE", "values": [round(float(overall.loc[m, "mae_percentage_points"]), 4) for m in METHODS]}, {"name": "RMSE", "values": [round(float(overall.loc[m, "rmse_percentage_points"]), 4) for m in METHODS]}], "y_label": "百分点"},
        ],
        "tables": [
            {"title": "训练与评估约定", "columns": ["集合", "目标时间", "每区域样本"], "rows": [["Train", "2022-09—12，先保留 168 小时作滞后历史", metadata["training_rows_per_region"]], ["Validation", "2023-01", 744], ["Test", "2023-02", 672]], "note": "按目标时间切分；Jan 仅诊断，不据测试集调整参数。"},
            {"title": "区域容量来自站点求和", "columns": ["TAZID", "充电桩数"], "rows": [[str(r), int(c)] for r, c in capacities.items()], "note": "inf.groupby('TAZID').charge_count.sum()；不覆盖同区多个站点。"},
        ],
        "method": ["固定作者源提交与 Git blob / SHA-256，按原始列序选择前 16 区域。", "将原始占用数量除以区域内所有站点充电桩数量之和。", "只使用过去 1/2/3/24/48/168 小时、滞后滚动均值与已知日历特征。", "固定单线程 HistGradientBoosting，与上一小时及昨日同小时基线比较。", "完整预测、区域级 MAE/RMSE、依赖与实测训练时间写入 results。"],
        "limitations": ["作者数据已进行异常替换及前后填充，原始缺失掩码不可恢复；不等于实时部署验证。", "README 的百分比表述与 CSV 数值存在口径差异；本工具依据作者归一化代码和公开容量边界校正。", "仅 16 个深圳区域与一个固定时间留出，不能推断其他城市或新站点效果。", "未使用 volume.csv，也未声称预测实测电功率、实现碳减排或接入真实充电站。"],
    }
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "data.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {m: overall.loc[m].to_dict() for m in METHODS}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"test": summary, "fit_seconds_total": metadata["fit_seconds_total"]}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="UrbanEV regional occupancy forecasting on CPU")
    parser.add_argument("command", choices=["demo", "download"], nargs="?", default="demo")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download(), indent=2))
    else:
        demo(args.output_dir)


if __name__ == "__main__":
    main()
