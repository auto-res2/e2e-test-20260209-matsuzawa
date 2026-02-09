import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import wandb
from scipy import stats


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def bootstrap_ci(data, n_samples=1000, alpha=0.05):
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_samples):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    lower = np.quantile(means, alpha / 2)
    upper = np.quantile(means, 1 - alpha / 2)
    return lower, upper


def permutation_pvalue(a, b, n_perm=1000):
    rng = np.random.default_rng(0)
    observed = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        new_a = combined[: len(a)]
        new_b = combined[len(a) :]
        diff = np.mean(new_a) - np.mean(new_b)
        if abs(diff) >= abs(observed):
            count += 1
    return count / n_perm


def metric_is_minimized(metric_name: str) -> bool:
    name = metric_name.lower()
    return any(k in name for k in ["loss", "perplexity", "error", "fragility", "cost", "calls", "rejection"])


def plot_learning_curve(history: pd.DataFrame, out_path: str, run_id: str) -> None:
    if "test_accuracy_running" not in history.columns:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(history.index, history["test_accuracy_running"], label="Test Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title(f"{run_id} Test Accuracy")
    if len(history) > 0:
        plt.annotate(
            f"{history['test_accuracy_running'].iloc[-1]:.3f}",
            (history.index[-1], history["test_accuracy_running"].iloc[-1]),
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_correctness_hist(correctness: List[int], out_path: str, run_id: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(correctness, bins=2, discrete=True)
    plt.xlabel("Correctness")
    plt.ylabel("Count")
    plt.title(f"{run_id} Per-question Correctness")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_accuracy_bar(run_ids: List[str], accs: List[float], cis: List[tuple], out_path: str):
    plt.figure(figsize=(8, 4))
    x = np.arange(len(run_ids))
    plt.bar(x, accs, yerr=[(a - c[0], c[1] - a) for a, c in zip(accs, cis)], capsize=4)
    plt.xticks(x, run_ids, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    for i, a in enumerate(accs):
        plt.text(i, a + 0.01, f"{a:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_bar(run_ids: List[str], values: List[float], metric_name: str, out_path: str):
    plt.figure(figsize=(8, 4))
    x = np.arange(len(run_ids))
    plt.bar(x, values)
    plt.xticks(x, run_ids, rotation=30, ha="right")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_table(rows: List[Dict[str, Any]], out_path: str):
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_demo_confusion_matrix(selected_demos: List[Dict[str, Any]], out_path: str, run_id: str):
    if not selected_demos:
        return
    y_true = []
    y_pred = []
    for d in selected_demos:
        gold = d.get("gold_answer")
        pred = d.get("answer")
        if gold is None or pred is None:
            continue
        y_true.append(int(str(gold).strip() == str(pred).strip()))
        y_pred.append(1 if str(gold).strip() == str(pred).strip() else 0)
    if not y_true:
        return
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Correctness")
    plt.ylabel("Actual Correctness")
    plt.title(f"{run_id} Demo Correctness Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str)
    args = parser.parse_args()

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]

    api = wandb.Api()
    run_ids = json.loads(args.run_ids)

    os.makedirs(args.results_dir, exist_ok=True)
    comparison_dir = os.path.join(args.results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    aggregated_metrics: Dict[str, Dict[str, float]] = {}
    per_run_correctness = {}
    accs, cis = [], []
    summary_map = {}
    generated_paths = []
    metric_rows = []

    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")
        history = run.history()
        summary = run.summary._json_dict
        config = dict(run.config)
        summary_map[run_id] = summary

        run_dir = os.path.join(args.results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        metrics = {
            "history": history.to_dict(orient="list"),
            "summary": summary,
            "config": config,
        }
        metrics_path = os.path.join(run_dir, "metrics.json")
        save_json(metrics_path, metrics)
        generated_paths.append(metrics_path)

        curve_path = os.path.join(run_dir, f"{run_id}_learning_curve.pdf")
        plot_learning_curve(history, curve_path, run_id)
        if os.path.exists(curve_path):
            generated_paths.append(curve_path)

        correctness = summary.get("per_question_correctness", [])
        if correctness:
            hist_path = os.path.join(run_dir, f"{run_id}_correctness_hist.pdf")
            plot_correctness_hist(correctness, hist_path, run_id)
            generated_paths.append(hist_path)
            per_run_correctness[run_id] = correctness

        demo_cm_path = os.path.join(run_dir, f"{run_id}_demo_confusion_matrix.pdf")
        plot_demo_confusion_matrix(summary.get("selected_demos", []), demo_cm_path, run_id)
        if os.path.exists(demo_cm_path):
            generated_paths.append(demo_cm_path)

        for k, v in summary.items():
            if isinstance(v, (int, float)):
                aggregated_metrics.setdefault(k, {})[run_id] = float(v)

        if "accuracy" in summary:
            accs.append(summary["accuracy"])
            ci = bootstrap_ci(correctness) if correctness else (summary["accuracy"], summary["accuracy"])
            cis.append(ci)

        metric_rows.append({
            "run_id": run_id,
            "accuracy": summary.get("accuracy", None),
            "demo_correctness_rate": summary.get("demo_correctness_rate", None),
            "avg_llm_calls_per_accepted_demo": summary.get("avg_llm_calls_per_accepted_demo", None),
            "paraphrase_fragility_selected_demos": summary.get("paraphrase_fragility_selected_demos", None),
            "rejection_rate": summary.get("rejection_rate", None),
        })

    primary_metric = "accuracy"
    best_proposed = {"run_id": None, "value": -1e9}
    best_baseline = {"run_id": None, "value": -1e9}
    for run_id, summary in summary_map.items():
        value = summary.get(primary_metric, None)
        if value is None:
            continue
        if "proposed" in run_id and value > best_proposed["value"]:
            best_proposed = {"run_id": run_id, "value": value}
        if ("baseline" in run_id or "comparative" in run_id) and value > best_baseline["value"]:
            best_baseline = {"run_id": run_id, "value": value}

    gap = None
    if best_proposed["run_id"] and best_baseline["run_id"] and best_baseline["value"] != 0:
        gap_raw = (best_proposed["value"] - best_baseline["value"]) / best_baseline["value"] * 100
        gap = gap_raw if not metric_is_minimized(primary_metric) else -gap_raw

    aggregated = {
        "primary_metric": "accuracy",
        "metrics": aggregated_metrics,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }

    aggregated_path = os.path.join(comparison_dir, "aggregated_metrics.json")
    save_json(aggregated_path, aggregated)
    generated_paths.append(aggregated_path)

    if accs:
        acc_plot_path = os.path.join(comparison_dir, "comparison_accuracy_bar_chart.pdf")
        plot_accuracy_bar(run_ids, accs, cis, acc_plot_path)
        generated_paths.append(acc_plot_path)

    for metric_name in [
        "demo_correctness_rate",
        "avg_llm_calls_per_accepted_demo",
        "paraphrase_fragility_selected_demos",
        "rejection_rate",
    ]:
        values = [summary_map[r].get(metric_name, None) for r in run_ids]
        if all(v is not None for v in values):
            out_path = os.path.join(comparison_dir, f"comparison_{metric_name}_bar_chart.pdf")
            plot_metric_bar(run_ids, values, metric_name, out_path)
            generated_paths.append(out_path)

    table_path = os.path.join(comparison_dir, "comparison_metrics_table.pdf")
    plot_metric_table(metric_rows, table_path)
    generated_paths.append(table_path)

    if best_proposed["run_id"] in per_run_correctness and best_baseline["run_id"] in per_run_correctness:
        pvalue = permutation_pvalue(
            np.array(per_run_correctness[best_proposed["run_id"]]),
            np.array(per_run_correctness[best_baseline["run_id"]]),
            n_perm=1000,
        )
        sig_path = os.path.join(comparison_dir, "comparison_significance.json")
        save_json(sig_path, {"pvalue": pvalue})
        generated_paths.append(sig_path)

    for p in generated_paths:
        print(p)


if __name__ == "__main__":
    main()
