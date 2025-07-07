import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io import load_json, save_json
from utils.metrics_tracker import auto_collect_anchors, collect_all_models_metrics, plot_metric_comparison
from config.settings import PROMPT_REFINER_MODEL

# === Auto-collect anchors / timestamps ===
anchors = auto_collect_anchors()
save_json("model_runs.json", anchors)
    
print(f"✅ Auto-collected anchor map: \n{json.dumps(anchors, indent=2)}")

# === Load model anchor timestamp ===
MODEL_RUNS = load_json("model_runs.json")
    
# === Choose REASON version ===
REASON = "with_reason" # or "without_reason"
BASE_RESULT_PATH = f"../../test_results/agent_results/{PROMPT_REFINER_MODEL}"

# === Collect metrics ===
df = collect_all_models_metrics(BASE_RESULT_PATH, MODEL_RUNS, REASON)
print(df)

# === Plot ===
plot_metric_comparison(df, metric="accuracy", reason=REASON)
plot_metric_comparison(df, metric="balanced_accuracy", reason=REASON)
plot_metric_comparison(df, metric="mcc", reason=REASON)

print(f"✅ Plots done for {REASON}")