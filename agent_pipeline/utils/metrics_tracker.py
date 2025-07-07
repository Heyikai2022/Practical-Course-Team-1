import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from utils.io import load_json
from config.settings import PROMPT_REFINER_MODEL

def auto_collect_anchors(base_result_path=f"../../test_results/agent_results/{PROMPT_REFINER_MODEL}"):
    anchors = defaultdict(dict)
    reason_pattern = re.compile(r"^(?P<timestamp>.+)_(?P<reason>with_reason|without_reason)_base$")
    
    for model_name in os.listdir(base_result_path):
        model_dir = os.path.join(base_result_path, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        for folder in os.listdir(model_dir):
            match = reason_pattern.match(folder)
            if match:
                timestamp = match.group("timestamp")
                reason = match.group("reason")
                anchors[model_name][reason] = timestamp
                
    return dict(anchors)

def collect_all_models_metrics(base_result_path, model_runs, reason):
    all_records = []
    
    for model_name, timestamps in model_runs.items():
        model_result_dir = os.path.join(base_result_path, model_name)
        anchor_prefix = f"{timestamps[reason]}_{reason}"

        print(f"🔍 Checking model: {model_name}")
        print(f"  Expected prefix: {anchor_prefix}")
        print(f"  Result dir: {model_result_dir}")

        if not os.path.exists(model_result_dir):
            print(f"❌ Model result folder does not exist: {model_result_dir}")
            continue

        found = False

        for folder in sorted(os.listdir(model_result_dir)):
            print(f"  📁 Found: {folder}")

            if folder.startswith(anchor_prefix):
                found = True
                print(f"    ✅ MATCH: {folder}")

                summary_path = os.path.join(model_result_dir, folder, "refined_summary.json")
                if not os.path.exists(summary_path):
                    summary_path = os.path.join(model_result_dir, folder, "summary.json")

                if os.path.exists(summary_path):
                    print(f"    ✅ Found summary: {summary_path}")
                    summary = load_json(summary_path)

                    if "_loop" in folder:
                        loop = int(folder.split("_loop")[-1])
                    else:
                        loop = 0

                    print(f"    ➡️ Adding record: model={model_name}, loop={loop}")

                    all_records.append({
                        "model": model_name,
                        "reason": reason,
                        "loop": loop,
                        "accuracy": summary.get("accuracy", 0),
                        "balanced_accuracy": summary.get("balanced_accuracy", 0),
                        "mcc": summary.get("mcc", 0)
                    })
                else:
                    print(f"    ❌ No summary found in: {folder}")

    if not found:
        print(f"❌ No folder matched prefix: {anchor_prefix}")


    if not all_records:
        print("⚠️ No records found — check timestamps, folder names, or summaries!")

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.sort_values(["model", "loop"])
    else:
        print("⚠️ DataFrame is EMPTY — returning empty DataFrame.")

    return df

def plot_metric_comparison(df, metric="accuracy", reason="with_reason", output_file=None):
    plt.figure(figsize=(10, 6))
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        plt.plot(subset["loop"], subset[metric], marker='o', label=model)
    
    reason_label = reason.replace('_', ' ').capitalize()
    plt.title(f"{metric.capitalize()} Trend Over Refinement Loops\n"
            f"Using {PROMPT_REFINER_MODEL} ({reason_label})")
    
    plt.xlabel("Refinement Loop")
    plt.ylabel(metric.capitalize())
    plt.xticks(sorted(df["loop"].unique()))
    if metric == "mcc":
        plt.ylim(-1, 1)
    else:
        plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    
    if output_file is None:
        output_file = f"../visualizations/core_{PROMPT_REFINER_MODEL}_{metric}_trend_{reason}.png"
        
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✅ Saved: {output_file}")
    
def plot_reason_comparison(df, metric="accuracy", output_file=None):
    if df.empty or "model" not in df.columns or "reason" not in df.columns:
        print(f"⚠️ No data for {metric} reason comparison.")
        return
    plt.figure(figsize=(10, 6))
    
    COLORS = ["blue", "green", "red", "purple", "orange", "brown"]
    models = sorted(df["model"].unique())
    color_map = {model: COLORS[i % len(COLORS)] for i, model in enumerate(models)}
    
    for model in df["model"].unique():
        for reason in df["reason"].unique():
            subset = df[(df["model"] == model) & (df["reason"] == reason)]
            if subset.empty:
                continue
        
            linestyle = "-" if reason == "with_reason" else "--"
            color = color_map[model]
            
            plt.plot(
                subset["loop"],
                subset[metric],
                linestyle, 
                marker="o", 
                color=color,
                label=f"{model} ({reason})"
            )
            
    plt.title(f"{metric.capitalize()} Trend Using {PROMPT_REFINER_MODEL}: \n"
            f"With vs Without Reason (All Models)")
    
    plt.xlabel("Refinemnt Loop")
    plt.ylabel(metric.capitalize())
    if metric == "mcc":
        plt.ylim(-1, 1)
    else:
        plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    
    if output_file is None:
        output_file = f"../visualizations/reason_comparison/core_{PROMPT_REFINER_MODEL}_{metric}_trend.png"
    
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✅ Saved: {metric}_reason_comparison.png")