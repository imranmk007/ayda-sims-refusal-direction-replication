"""
Inspect saved checkpoint results from txt-experiment-test.py.
Run from the same directory as the experiment:
    python inspect_results.py
or point to a different results dir:
    python inspect_results.py --dir /path/to/results_test
"""

import argparse
import json
import os
import sys

import torch

STEPS = {
    1: "step1_harmful_text_results.pt",
    2: None,  # no response file, just vectors
    3: "step3_harmful_image_baseline_results.pt",
    5: "step5_ablated_harmful_image_results.pt",
    6: "step6_addition_results.pt",
    7: "step7_safe_image_results.pt",
    8: "step8_summary_statistics.json",
}

LAYER_START = 13
LAYER_END = 31


def hr(char="-", width=80):
    print(char * width)


def show_responses(queries, responses, judgments, label, n=None):
    hr("=")
    print(f"  {label}")
    hr("=")
    items = list(zip(queries or [""] * len(responses), responses, judgments))
    if n:
        items = items[:n]
    for i, (q, r, j) in enumerate(items):
        verdict = "REFUSAL" if j is True else ("COMPLIANCE" if j is False else "UNKNOWN")
        hr()
        print(f"  [{i}] {verdict}")
        if q:
            print(f"  QUERY   : {q[:120]}")
        print(f"  RESPONSE: {r[:600]}")
    hr()
    refused = sum(1 for j in judgments if j is True)
    total = len(judgments)
    print(f"  Refusal rate: {refused}/{total} = {refused/total:.1%}")
    print()


def inspect_step1(d):
    path = os.path.join(d, STEPS[1])
    if not os.path.exists(path):
        print("Step 1 results not found yet.\n")
        return
    data = torch.load(path, map_location="cpu")
    print("\n===== STEP 1: Harmful Text Inference =====")
    print(f"  Total inferred : {data.get('total_inferred', '?')}")
    print(f"  Refused (kept) : {data.get('num_refused', '?')}")
    print(f"  Refusal rate   : {data.get('refusal_rate', 0):.1%}")
    print(f"  Refused indices: {data.get('refused_indices', [])}")
    print()

    queries = data.get("all_queries", [])
    refused_indices = set(data.get("refused_indices", []))
    responses = data.get("responses", [])  # only refused responses stored here

    # Reconstruct per-sample view from refused indices + responses
    print("  Refused responses (used to build refusal vector):")
    for i, (idx, r) in enumerate(zip(sorted(refused_indices), responses)):
        q = queries[idx] if idx < len(queries) else ""
        hr()
        print(f"  [{idx}] REFUSED")
        print(f"  QUERY   : {q[:120]}")
        print(f"  RESPONSE: {r[:600]}")
    hr()


def inspect_step3(d):
    path = os.path.join(d, STEPS[3])
    if not os.path.exists(path):
        print("Step 3 results not found yet.\n")
        return
    data = torch.load(path, map_location="cpu")
    print("\n===== STEP 3: Harmful Image Baseline (FigStep) =====")
    show_responses(
        queries=None,
        responses=data["all_responses"],
        judgments=data["judgments"],
        label="Harmful image baseline",
    )


def inspect_step5(d):
    path = os.path.join(d, STEPS[5])
    if not os.path.exists(path):
        print("Step 5 results not found yet.\n")
        return
    data = torch.load(path, map_location="cpu")
    print("\n===== STEP 5: Ablated Harmful Image Inference =====")
    show_responses(
        queries=None,
        responses=data["responses"],
        judgments=data["judgments"],
        label=f"Ablated (jailbreak) | baseline={data['baseline_refusal_rate']:.1%} → ablated={data['ablated_refusal_rate']:.1%}",
    )


def inspect_step6(d):
    path = os.path.join(d, STEPS[6])
    if not os.path.exists(path):
        print("Step 6 results not found yet.\n")
        return
    data = torch.load(path, map_location="cpu")
    print("\n===== STEP 6: Addition Inference =====")
    ha = data["harmful_added"]
    show_responses(
        queries=None,
        responses=ha["responses"],
        judgments=ha["judgments"],
        label=f"Harmful images + vector added | baseline={ha['baseline_refusal_rate']:.1%} → added={ha['refusal_rate']:.1%}",
    )
    sa = data["safe_added"]
    show_responses(
        queries=None,
        responses=sa["responses"],
        judgments=sa["judgments"],
        label=f"Safe images + vector added | rate={sa['refusal_rate']:.1%} (want ~100%)",
    )


def inspect_step7(d):
    path = os.path.join(d, STEPS[7])
    if not os.path.exists(path):
        print("Step 7 results not found yet.\n")
        return
    data = torch.load(path, map_location="cpu")
    print("\n===== STEP 7: Safe Image KL Data =====")
    show_responses(
        queries=None,
        responses=data["safe_responses"],
        judgments=data["safe_judgments"],
        label=f"Safe images baseline | rate={data['safe_refusal_rate']:.1%} (want ~0%)",
    )
    show_responses(
        queries=None,
        responses=data["ablated_responses"],
        judgments=data["ablated_judgments"],
        label=f"Safe images ablated | rate={data['ablated_refusal_rate']:.1%}",
    )


def inspect_step8(d):
    path = os.path.join(d, STEPS[8])
    if not os.path.exists(path):
        print("Step 8 results not found yet.\n")
        return
    with open(path) as f:
        stats = json.load(f)
    print("\n===== STEP 8: Summary Statistics =====")
    print(json.dumps(stats["refusal_rates"], indent=4))
    print()
    print("  KL divergence:")
    print(json.dumps(stats["kl_divergence"], indent=4))
    print()
    pl = stats["peak_layer"]
    print(f"  Peak layer: {pl['layer']} (mean dot product = {pl['mean_dot_product']:.4f})")


def inspect_vectors(d):
    path = os.path.join(d, "step2_refusal_vectors.pt")
    if not os.path.exists(path):
        print("Step 2 refusal vectors not found yet.\n")
        return
    vecs = torch.load(path, map_location="cpu")
    print("\n===== STEP 2: Refusal Vector Norms =====")
    for layer_idx in range(LAYER_START, LAYER_END + 1):
        if layer_idx in vecs:
            norm = vecs[layer_idx].norm().item()
            print(f"  Layer {layer_idx:2d}: norm = {norm:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results_test", help="results directory")
    parser.add_argument("--steps", default="1,2,3,5,6,7,8", help="comma-separated step numbers to inspect")
    args = parser.parse_args()

    d = args.dir
    if not os.path.isdir(d):
        print(f"Results directory not found: {d}")
        sys.exit(1)

    steps = {int(s) for s in args.steps.split(",")}

    if 1 in steps:
        inspect_step1(d)
    if 2 in steps:
        inspect_vectors(d)
    if 3 in steps:
        inspect_step3(d)
    if 5 in steps:
        inspect_step5(d)
    if 6 in steps:
        inspect_step6(d)
    if 7 in steps:
        inspect_step7(d)
    if 8 in steps:
        inspect_step8(d)


if __name__ == "__main__":
    main()
