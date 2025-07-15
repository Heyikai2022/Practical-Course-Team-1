import json
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_json, save_json
from utils.metrics_tracker import auto_collect_anchors, collect_all_models_metrics, plot_metric_comparison, plot_reason_comparison
from config.settings import PROMPT_REFINER_MODEL

# === Auto-collect anchors / timestamps ===
anchors = auto_collect_anchors()
save_json("model_runs.json", anchors)
    
print(f"✅ Auto-collected anchor map: \n{json.dumps(anchors, indent=2)}")

# === Load model anchor timestamp ===
MODEL_RUNS = load_json("model_runs.json")
    
BASE_RESULT_PATH = f"../../test_results/agent_results/{PROMPT_REFINER_MODEL}"

# === Collect both versions ===
df_with_reason = collect_all_models_metrics(BASE_RESULT_PATH, MODEL_RUNS, reason="with_reason")
df_without_reason = collect_all_models_metrics(BASE_RESULT_PATH, MODEL_RUNS, reason="without_reason")

# Combine into one df
df = None
if not df_with_reason.empty and not df_without_reason.empty:
    df = pd.concat([df_with_reason, df_without_reason], ignore_index=True)
elif not df_with_reason.empty:
    df = df_with_reason
elif not df_without_reason.empty:
    df = df_without_reason
else:
    df = pd.DataFrame()

print(df)

# === Plot with and without reason separately ===
plot_metric_comparison(df_with_reason, metric="accuracy", reason="with_reason")
plot_metric_comparison(df_without_reason, metric="accuracy", reason="without_reason")

plot_metric_comparison(df_with_reason, metric="balanced_accuracy", reason="with_reason")
plot_metric_comparison(df_without_reason, metric="balanced_accuracy", reason="without_reason")

plot_metric_comparison(df_with_reason, metric="mcc", reason="with_reason")
plot_metric_comparison(df_without_reason, metric="mcc", reason="without_reason")

# === Plot reason comparison: both with and without reason at once ===
plot_reason_comparison(df, metric="accuracy")
plot_reason_comparison(df, metric="balanced_accuracy")
plot_reason_comparison(df, metric="mcc")

print(f"✅ All plots done for using {PROMPT_REFINER_MODEL} as core model.")