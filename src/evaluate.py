import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
import yaml
from scipy import stats
from sklearn.metrics import confusion_matrix

PRIMARY_METRIC = "val_accuracy"

################################################################################
#                         Load global WandB credentials                        #
################################################################################

def _wandb_creds() -> Dict[str, str]:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    return {"entity": base["wandb"]["entity"], "project": base["wandb"]["project"]}

################################################################################
#                             Per-run export helper                            #
################################################################################

def _export_run(run: wandb.apis.public.Run, out_dir: Path):
    rid = run.config.get("run", {}).get("run_id", run.id)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = run.history(keys=["train_loss", PRIMARY_METRIC, "carbon_used_g"], samples=10_000)
    summary = dict(run.summary)
    cfg = dict(run.config)

    # Dump raw data ----------------------------------------------------------
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": history.to_dict("list"), "summary": summary, "config": cfg}, f, indent=2)

    # Learning curve ---------------------------------------------------------
    plt.figure(figsize=(6, 4))
    if "train_loss" in history.columns:
        sns.lineplot(x=history.index, y=history["train_loss"], label="train_loss")
    if PRIMARY_METRIC in history.columns:
        sns.lineplot(x=history.index, y=history[PRIMARY_METRIC], label=PRIMARY_METRIC)
    plt.xlabel("step"); plt.ylabel("value"); plt.title(f"{rid} – learning curve"); plt.legend(); plt.tight_layout()
    path_lc = out_dir / f"{rid}_learning_curve.pdf"; plt.savefig(path_lc); plt.close(); print(path_lc)

    # Confusion matrix if provided -----------------------------------------
    preds, labels = summary.get("val_predictions"), summary.get("val_labels")
    if preds and labels:
        uniq = sorted(set(preds + labels))[:100]
        cm = confusion_matrix(labels, preds, labels=uniq)
        plt.figure(figsize=(5, 4)); sns.heatmap(cm, annot=False, cmap="Blues"); plt.xlabel("pred"); plt.ylabel("true"); plt.tight_layout()
        cm_path = out_dir / f"{rid}_confusion_matrix.pdf"; plt.savefig(cm_path); plt.close(); print(cm_path)

################################################################################
#                           Aggregated comparison                              #
################################################################################

def _aggregate(runs: Dict[str, wandb.apis.public.Run], results_dir: Path):
    cmp_dir = results_dir / "comparison"; cmp_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Dict[str, float]] = {}
    for rid, run in runs.items():
        uid = run.config.get("run", {}).get("run_id", rid)
        for k, v in run.summary.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics.setdefault(k, {})[uid] = float(v)

    if PRIMARY_METRIC not in metrics:
        raise RuntimeError(f"Primary metric '{PRIMARY_METRIC}' not found in any run summaries")

    proposed = {k: v for k, v in metrics[PRIMARY_METRIC].items() if "proposed" in k.lower() or "c3po" in k.lower()}
    baseline = {k: v for k, v in metrics[PRIMARY_METRIC].items() if any(s in k.lower() for s in ["baseline", "comparative", "blur"])}
    best_prop_id = max(proposed, key=proposed.get) if proposed else max(metrics[PRIMARY_METRIC], key=metrics[PRIMARY_METRIC].get)
    best_base_id = max(baseline, key=baseline.get) if baseline else best_prop_id
    gap = (metrics[PRIMARY_METRIC][best_prop_id] - metrics[PRIMARY_METRIC][best_base_id]) / max(1e-9, metrics[PRIMARY_METRIC][best_base_id]) * 100.0

    aggregated = {
        "primary_metric": "(A) GSM1k accuracy at 120 g CO₂;  (B) area-under-accuracy-vs-CO₂ Pareto curve domination count.",
        "metrics": metrics,
        "best_proposed": {"run_id": best_prop_id, "value": metrics[PRIMARY_METRIC][best_prop_id]},
        "best_baseline": {"run_id": best_base_id, "value": metrics[PRIMARY_METRIC][best_base_id]},
        "gap": gap,
    }
    with open(cmp_dir / "aggregated_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    # Metrics table ----------------------------------------------------------
    pd.DataFrame(metrics).to_csv(cmp_dir / "metrics_table.csv")

    # Bar chart --------------------------------------------------------------
    plt.figure(figsize=(8, 4))
    order = list(metrics[PRIMARY_METRIC].keys())
    sns.barplot(x=order, y=[metrics[PRIMARY_METRIC][k] for k in order], palette="viridis")
    plt.xticks(rotation=45, ha="right"); plt.ylabel(PRIMARY_METRIC); plt.tight_layout()
    bar_path = cmp_dir / "comparison_primary_metric_bar_chart.pdf"; plt.savefig(bar_path); plt.close(); print(bar_path)

    # Box plot ---------------------------------------------------------------
    df_box = pd.DataFrame({"run_id": list(metrics[PRIMARY_METRIC].keys()), PRIMARY_METRIC: list(metrics[PRIMARY_METRIC].values())})
    df_box["group"] = df_box.run_id.str.contains("proposed|c3po", case=False).map({True: "proposed", False: "baseline"})
    plt.figure(figsize=(4, 4)); sns.boxplot(data=df_box, x="group", y=PRIMARY_METRIC); sns.stripplot(data=df_box, x="group", y=PRIMARY_METRIC, color="black", size=5)
    plt.tight_layout(); box_path = cmp_dir / "comparison_primary_metric_box_plot.pdf"; plt.savefig(box_path); plt.close(); print(box_path)

    # Welch t-test -----------------------------------------------------------
    prop_vals, base_vals = [v for v in proposed.values()], [v for v in baseline.values()]
    if prop_vals and base_vals:
        t, p = stats.ttest_ind(prop_vals, base_vals, equal_var=False)
        with open(cmp_dir / "significance_tests.json", "w", encoding="utf-8") as f:
            json.dump({"t_stat": float(t), "p_value": float(p), "significant_(α=0.05)": bool(p < 0.05)}, f, indent=2)

################################################################################
#                                       CLI                                   #
################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help='JSON list e.g. "[\"run-1\", \"run-2\"]"')
    args = parser.parse_args()

    run_list: List[str] = json.loads(args.run_ids)
    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)

    creds = _wandb_creds(); api = wandb.Api()
    runs = {rid: api.run(f"{creds['entity']}/{creds['project']}/{rid}") for rid in run_list}

    # Per-run processing ------------------------------------------------------
    for rid, run in runs.items():
        _export_run(run, results_dir / rid)

    # Aggregated view ---------------------------------------------------------
    _aggregate(runs, results_dir)

if __name__ == "__main__":
    main()