import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.prompt_refiner import get_refinement_chain
from agents.target_model import call_target_model
from utils.evaluation import parse_judgment, compare, compute_metrics
from utils.io import load_json, save_json

from config.settings import PROMPT_REFINER_MODEL, TARGET_MODEL_NAME, REASON

# === Link back to baseline run ===
BASELINE_TIMESTAMP = "20250706_200201"  # Fill the baseline_timestamp (to be found in the summary.json) here to track the whole refinement, e.g. the baseline run: 20240627_223000
LOOP = 5  # Which loop number this is, should now be manually edited (could be automated later!)

reason_suffix = "with_reason" if REASON else "without_reason"
model_name = TARGET_MODEL_NAME.split("/")[-1]

# === Build paths ===
if LOOP == 1:
    previous_dir = f"../../test_results/agent_results/{PROMPT_REFINER_MODEL}/{model_name}/{BASELINE_TIMESTAMP}_{reason_suffix}_base"
else:
    previous_dir = f"../../test_results/agent_results/{PROMPT_REFINER_MODEL}/{model_name}/{BASELINE_TIMESTAMP}_{reason_suffix}_loop{LOOP - 1}"

result_dir = f"../../test_results/agent_results/{PROMPT_REFINER_MODEL}/{model_name}/{BASELINE_TIMESTAMP}_{reason_suffix}_loop{LOOP}"
os.makedirs(result_dir, exist_ok=True)

# === Load previous alignments ===
try:
    prev_alignment = load_json(f"{previous_dir}/alignment.json")
except FileNotFoundError:
    prev_alignment = []

# === Load the misaligned samples ===
fp = load_json(f"{previous_dir}/fp.json")
fn = load_json(f"{previous_dir}/fn.json")
misses = fp + fn

# Load prev alignment samples:
prev_alignment = load_json(f"{previous_dir}/alignment.json") \
    if LOOP == 1 \
    else load_json(f"{previous_dir}/all_alignment.json")

# === Set up ===
refiner = get_refinement_chain()
log_prompts = []
fp_new = []
fn_new = []
new_alignment = []

for miss in tqdm(misses, desc=f"Refinement loop {LOOP}"):
    improved_prompt = refiner.run({
        "neutral_persona": miss["neutral_persona"],
        "original_prompt": miss["prompt"],
        "original_output": miss["response"]
    })
    
    new_output = call_target_model(improved_prompt)
    new_judgment = parse_judgment(new_output)
    new_outcome = compare(new_judgment, miss["label"])

    log_prompts.append({
        "id": miss["id"],
        "refined_prompt": improved_prompt,
        "new_output": new_output,
        "new_judgment": new_judgment,
        "new_outcome": new_outcome
    })
    
    if new_outcome == "alignment":
        new_alignment.append(miss)
    elif new_outcome == "fp":
        fp_new.append(miss)
    elif new_outcome == "fn":
        fn_new.append(miss)
        
# Combine all alignments so far
all_alignment = prev_alignment + new_alignment

save_json(f"{result_dir}/refined_results.json", log_prompts)
save_json(f"{result_dir}/fp.json", fp_new)
save_json(f"{result_dir}/fn.json", fn_new)
save_json(f"{result_dir}/all_alignment.json", all_alignment)

# === Compute metrics ===
metrics = compute_metrics(all_alignment, fp_new, fn_new)

summary = {
    "baseline": BASELINE_TIMESTAMP,
    "loop": LOOP,
    "reason": reason_suffix,
    "refined": len(log_prompts),
    "new_alignment": len(new_alignment), 
    "all_alignment": len(all_alignment),
    "fp": len(fp_new), 
    "fn": len(fn_new), 
    **metrics
}

save_json(f"{result_dir}/refined_summary.json", summary)

print(f"✅ Refinement loop {LOOP} done: {len(log_prompts)} refined.")
print(f"  ✔️ Accuracy: {metrics['accuracy']:.4f}")
print(f"  ✔️ Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
print(f"  ✔️ MCC: {metrics['mcc']:.4f}")
print(f" Alignments this loop: {len(new_alignment)}")
print(f" Remaining FP: {len(fp_new)} | FN: {len(fn_new)}")
print(f"📁 Results saved to: {result_dir}")