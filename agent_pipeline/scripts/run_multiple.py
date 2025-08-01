import os
import sys
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.target_model import call_target_model
from utils.evaluation import parse_judgment, compare, compute_metrics
from utils.io import load_json, save_json
from config.settings import TARGET_MODEL_NAME, REASON, DATASET_PATH

# === Configs ===
MODEL_NAME = TARGET_MODEL_NAME.split("/")[-1]
REASON_SUFFIX = "with_reason" if REASON else "without_reason"
NUM_RUNS = 10
PROMPT_KEY = "prompt_with_reason" if REASON else "prompt_without_reason"
DATA = load_json(DATASET_PATH)

# === Output directory ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_base = f"../../test_results/stat_eval/{MODEL_NAME}/{REASON_SUFFIX}/{timestamp}"
os.makedirs(result_base, exist_ok=True)

# === Run statistical prompting ===
for seed in range(NUM_RUNS):
    print(f"\n🚀 Running seed {seed} | model: {MODEL_NAME} | prompt: {REASON_SUFFIX}")
    run_dir = os.path.join(result_base, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    results = []
    for entry in tqdm(DATA, desc=f"Seed {seed}"):
        prompt = entry[PROMPT_KEY]
        label = entry["label"]
        try:
            output = call_target_model(prompt)
            judgment = parse_judgment(output)
            outcome = compare(judgment, label)

            results.append({
                "id": entry["id"],
                "prompt": prompt,
                "label": label,
                "response": output,
                "judgment": judgment,
                "outcome": outcome
            })
        except Exception as e:
            print(f"❌ Sample {entry['id']} failed: {e}")

    save_json(f"{run_dir}/results.json", results)

    # Save per-run metrics
    alignment = [r for r in results if r["outcome"] == "alignment"]
    fp = [r for r in results if r["outcome"] == "fp"]
    fn = [r for r in results if r["outcome"] == "fn"]
    metrics = compute_metrics(alignment, fp, fn)

    summary = {
        "model": MODEL_NAME,
        "reason": REASON_SUFFIX,
        "seed": seed,
        "alignments": len(alignment),
        "fp": len(fp),
        "fn": len(fn),
        **metrics,
        "timestamp": timestamp
    }
    save_json(f"{run_dir}/summary.json", summary)
    print(f"✅ Saved results and summary to {run_dir}")