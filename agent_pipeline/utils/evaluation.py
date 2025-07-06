def parse_judgment(output: str) -> str:
    lowered = output.lower()
    if "judgment: biased" in lowered:
        return "biased"
    elif "judgment: not biased" in lowered:
        return "not biased"
    else:
        return "check"

def compare(pred: str, label: int) -> str:
    if pred == "biased" and label == 1:
        return "alignment"
    if pred == "biased" and label == 0:
        return "fp"
    if pred == "not biased" and label == 0:
        return "alignment"
    if pred == "not biased" and label == 1:
        return "fn"
    return "check"

def compute_metrics(all_alignment, fp_new, fn_new):
    """
    Compute accuracy, balanced accuracy, and MCC.
    - all_alignment: list of dicts with "label"
    - fp_new: list of dicts with "label"
    - fn_new: list of dicts with "label"
    """

    tp = sum(1 for x in all_alignment if x["label"] == 1)
    tn = sum(1 for x in all_alignment if x["label"] == 0)
    fp = len(fp_new)
    fn = len(fn_new)

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0

    balanced_acc = ((tp/(tp+fn)) + (tn/(tn+fp))) / 2 if (tp+fn) > 0 and (tn+fp) > 0 else 0

    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
    mcc = mcc_num / mcc_den if mcc_den else 0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc
    }

