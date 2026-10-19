"""Descriptive utility analysis and chronological held-out prediction."""

import argparse
import json
from pathlib import Path

FEATURES = ["delay", "query_entropy", "return_entropy", "prediction_changed",
            "prediction_js", "feature_temporal_cosine", "feature_batch_cosine",
            "gradient_temporal_cosine", "gradient_reference_cosine"]


def analyze(output):
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    frames = []
    for delay in manifest["delays"]:
        folder = output / f"D{delay}"
        result = json.loads((folder / "result.json").read_text())
        if result["status"] != "complete":
            raise ValueError(f"Incomplete experiment: {folder}")
        frame = pd.read_csv(folder / "utilities.csv")
        frame["complete_window"] = frame["complete_window"].astype(str).str.lower() == "true"
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    full = data[data.complete_window & data.utility.notna()].copy()
    summaries = []
    for delay in manifest["delays"]:
        group = full[full.delay == delay]
        total = data[data.delay == delay]
        summaries.append({"delay": delay, "returned_events": len(total),
                          "complete_events": len(group),
                          "positive_fraction": float((group.utility > 1e-6).mean()) if len(group) else None,
                          "negative_fraction": float((group.utility < -1e-6).mean()) if len(group) else None,
                          "zero_fraction": float((group.utility.abs() <= 1e-6).mean()) if len(group) else None,
                          "mean_utility_pp": float(group.utility.mean() * 100) if len(group) else None,
                          "median_utility_pp": float(group.utility.median() * 100) if len(group) else None})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    correlations = []
    for scope, subset in [("all", full)] + [(f"D{d}", full[full.delay == d]) for d in manifest["delays"]]:
        for feature in FEATURES:
            value = None
            if len(subset) > 2 and subset[feature].nunique() > 1 and subset.utility.nunique() > 1:
                value = float(subset[feature].corr(subset.utility, method="spearman"))
            correlations.append({"scope": scope, "feature": feature,
                                 "events": len(subset), "spearman_utility": value})
    pd.DataFrame(correlations).to_csv(output / "feature_correlations.csv", index=False)

    # Use one global time boundary across delays. Embargo overlapping utility
    # windows; never split labels from one return batch across train and test.
    times = sorted(full.return_step.unique())
    prediction = {"status": "insufficient_data", "note": "Need chronological train/test groups after horizon embargo"}
    if len(times) >= 4:
        boundary = int(times[min(len(times) - 1, int(len(times) * 0.7))])
        train = full[full.return_step + manifest["horizon"] < boundary]
        test = full[full.return_step >= boundary]
        if len(train) >= 10 and len(test) >= 5:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            baseline = DummyRegressor(strategy="mean")
            x_train, x_test = train[FEATURES], test[FEATURES]
            model.fit(x_train, train.utility)
            baseline.fit(x_train, train.utility)
            estimates = model.predict(x_test)
            prediction = {
                "status": "complete", "model": "StandardScaler + Ridge(alpha=1)",
                "boundary_step": boundary, "train_events": len(train), "test_events": len(test),
                "embargo_batches": manifest["horizon"],
                "mae_pp": float(mean_absolute_error(test.utility, estimates) * 100),
                "baseline_mae_pp": float(mean_absolute_error(
                    test.utility, baseline.predict(x_test)) * 100),
                "r2": float(r2_score(test.utility, estimates)) if test.utility.nunique() > 1 else None,
                "positive_auc": float(roc_auc_score(test.utility > 1e-6, estimates))
                if (test.utility > 1e-6).nunique() == 2 else None,
                "sign_accuracy": float(np.mean((estimates > 1e-6) == (test.utility.to_numpy() > 1e-6))),
                "test_majority_sign_accuracy": float(max((test.utility > 1e-6).mean(),
                                                          (test.utility <= 1e-6).mean())),
            }
            heldout = test[["delay", "sample_id", "return_step", "utility"]].copy()
            heldout["predicted_utility"] = estimates
            heldout.to_csv(output / "heldout_predictions.csv", index=False)
            pd.DataFrame({"feature": FEATURES, "standardized_ridge_coefficient":
                          model[-1].coef_}).to_csv(output / "predictor_coefficients.csv", index=False)
    (output / "prediction_metrics.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    table = ["| 延迟(batch) | 返回标签 | 完整窗口 | 正收益比例 | 负收益比例 | 平均U(百分点) |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        def fmt(key, scale=1):
            return "N/A" if row[key] is None else f"{row[key] * scale:.4f}"
        table.append(f"| {row['delay']} | {row['returned_events']} | {row['complete_events']} | "
                     f"{fmt('positive_fraction', 100)}% | {fmt('negative_fraction', 100)}% | "
                     f"{fmt('mean_utility_pp')} |")
    (output / "summary.md").write_text(
        "# PACS 历史标签监督价值\n\n" + "\n".join(table)
        + "\n\nU 是相同未来 batch 上 USE−SKIP 的准确率差，按 batch 等权平均。"
        "仅完整窗口进入统计；阈值为准确率差 1e-6。"
        "不同延迟的查询样本可能不同，均值变化不能解释为同一标签的纯延迟因果效应。"
        "\n\n预测器使用返回时已可观察指标；按时间留出后30%的返回步，训练窗口与测试区间隔离。"
        "预测指标见 prediction_metrics.json，相关性不是因果证据。单seed结果是探索性结果。\n",
        encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", required=True, type=Path)
    analyze(cli.parse_args().output_dir)


if __name__ == "__main__":
    main()
