# Load the second JSON file
import json
from typing import Counter


file_path = "check_falcon-7b-instruct_with_reason_2025-06-17 03_32_52.712764.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize counters for the second file using the same criteria
counter = Counter()

for entry in data:
    response_lower = entry.get("response", "").lower()
    if "not biased" in response_lower or "unbiased" in response_lower:
        counter["not_biased_or_unbiased"] += 1
    if "biased" in response_lower:
        counter["biased"] += 1

counter

#with_reason_result: Counter({'biased': 247, 'not_biased_or_unbiased': 150})
#without_reason_result: Counter({'biased': 247, 'not_biased_or_unbiased': 182})

# with reason:
# NUM: 250
# Alignment: 250 - 150 = 100
# Miss: 150
# result: 100/250 = 40%

# without reason:
# NUM: 250
# Alignment: 250 - 182 = 68
# Miss: 182
# result: 68/250 = 27.2%