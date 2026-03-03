# log in to hugging face

from huggingface_hub import login

login()


# imports
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc
import json
import os
from datetime import datetime

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore


# define model and device
MODEL_PATH = "google/gemma-2b-it"
DEVICE = "mps"

model = HookedTransformer.from_pretrained(
    MODEL_PATH,
    device=DEVICE,
    dtype=torch.float16,
    default_padding_side="left",
)
model.tokenizer.padding_side = "left"

# download the two datasets and intialize them w/ the test-train split


def pull_harmful_instructions():
    response = requests.get(
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    )
    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = df["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def pull_harmless_instructions():
    dataset = load_dataset("tatsu-lab/alpaca")
    instructions = [
        ex["instruction"] for ex in dataset["train"] if ex["input"].strip() == ""
    ]
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


# IMPORTANT: These are the two datasets we'll be using throughout the code
harmful_inst_train, harmful_inst_test = pull_harmful_instructions()
harmless_inst_train, harmless_inst_test = pull_harmless_instructions()


# define the gemma chat template for the formatting
GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""


# create the tokenizing functions; i wasn't too sure how to do this part, so i asked claude
def tokenize_instructions(
    tokenizer: AutoTokenizer,
    instructions: List[str],
) -> Int[Tensor, "batch_size seq_len"]:
    prompts = [GEMMA_CHAT_TEMPLATE.format(instruction=instr) for instr in instructions]
    return tokenizer(
        prompts, padding=True, truncation=False, return_tensors="pt"
    ).input_ids


# encapsulates the tokenize_instructions() function
tokenize_instructions_fn = functools.partial(
    tokenize_instructions, tokenizer=model.tokenizer
)

# useful in getting the ending tokens after the <instruction> tag (just removes the input which is aesthetically better)
eoi_token_ids = model.tokenizer.encode(
    GEMMA_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False
)


# == REFUSAL DIRECTION == #

N_TRAIN = 32  # number of harmful-harmless pairs we use to find the refusal dir

harmful_train_subset = harmful_inst_train[:N_TRAIN]
harmless_train_subset = harmless_inst_train[:N_TRAIN]

harmful_toks = tokenize_instructions_fn(instructions=harmful_train_subset)
harmless_toks = tokenize_instructions_fn(instructions=harmless_train_subset)


# also had to use Claude Code for this part...

# gets the logits and activations on the QA pairs for harmful questions
harmful_logits, harmful_cache = model.run_with_cache(
    harmful_toks, names_filter=lambda name: "resid_pre" in name
)
harmless_logits, harmless_cache = model.run_with_cache(
    harmless_toks, names_filter=lambda name: "resid_pre" in name
)

n_layers = model.cfg.n_layers  # should be like 18 for gemma-2b-it?
d_model = model.cfg.d_model  # should be like 2048 for gemma-2b-it
n_positions = len(eoi_token_ids)

mean_diffs = torch.zeros(
    (n_positions, n_layers, d_model), dtype=torch.float32, device=DEVICE
)

# testing for each layer and position because we don't know which one will be the most effective
for pos_idx, neg_pos in enumerate(range(-n_positions, 0)):
    for layer in range(n_layers):
        harmful_mean = harmful_cache[("resid_pre", layer)][:, neg_pos, :].mean(dim=0)
        harmless_mean = harmless_cache[("resid_pre", layer)][:, neg_pos, :].mean(dim=0)
        mean_diffs[pos_idx, layer] = (harmful_mean - harmless_mean).float()

del harmful_cache, harmless_cache, harmful_logits, harmless_logits
gc.collect()
torch.mps.empty_cache() if DEVICE == "mps" else torch.cuda.empty_cache()

REFUSAL_TOK = 235285


def compute_refusal_score(logits, refusal_tok_id=REFUSAL_TOK, epsilon=1e-8):
    logits = logits.to(torch.float32)
    logits = logits[:, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_prob = probs[:, refusal_tok_id].mean()
    nonrefusal_prob = 1.0 - refusal_prob
    return torch.log(refusal_prob + epsilon) - torch.log(nonrefusal_prob + epsilon)


def ablation_hook(activation, hook, direction):
    direction_norm = direction / (direction.norm() + 1e-8)
    direction_norm = direction_norm.to(activation.dtype).to(activation.device)
    proj = einops.einsum(activation, direction_norm, "... d, d -> ...") * direction_norm
    return activation - proj


N_VAL = 8
harmful_val_subset = harmful_inst_train[N_TRAIN : N_TRAIN + N_VAL]
harmless_val_subset = harmless_inst_train[N_TRAIN : N_TRAIN + N_VAL]

harmful_val_toks = tokenize_instructions_fn(instructions=harmful_val_subset)
harmless_val_toks = tokenize_instructions_fn(instructions=harmless_val_subset)

baseline_harmful_logits = model(harmful_val_toks)
baseline_harmful_score = compute_refusal_score(baseline_harmful_logits)

ablation_scores = torch.zeros((n_positions, n_layers))
kl_scores = torch.zeros((n_positions, n_layers))

print("\nTesting candidates...")
for pos_idx in range(n_positions):
    for layer_idx in range(n_layers):
        direction = mean_diffs[pos_idx, layer_idx]
        direction_norm = direction / (direction.norm() + 1e-8)

        hook_fn = functools.partial(ablation_hook, direction=direction_norm)

        hook_points = []
        for l in range(n_layers):
            hook_points.append((utils.get_act_name("resid_pre", l), hook_fn))

        with model.hooks(fwd_hooks=hook_points):
            ablation_harmful_logits = model(harmful_val_toks)

        ablation_score = compute_refusal_score(ablation_harmful_logits.logits)
        ablation_scores[pos_idx, layer_idx] = ablation_score.item()

        if pos_idx % 2 == 0 and layer_idx % 5 == 0:
            print(
                f"  pos={pos_idx}, layer={layer_idx}: score={ablation_score.item():.4f}"
            )

best_pos_idx, best_layer = torch.where(ablation_scores == ablation_scores.min())
best_pos_idx = best_pos_idx[0].item()
best_layer = best_layer[0].item()

best_direction = mean_diffs[best_pos_idx, best_layer]
best_direction = best_direction / (best_direction.norm() + 1e-8)

best_neg_pos = -n_positions + best_pos_idx
print(f"\nSelected direction: pos={best_neg_pos}, layer={best_layer}")
print(f"  Baseline refusal score: {baseline_harmful_score.item():.4f}")
print(
    f"  Ablation refusal score: {ablation_scores[best_pos_idx, best_layer].item():.4f}"
)

# Save results
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
baseline_score = float(baseline_harmful_score.item())
ablation_score = float(ablation_scores[best_pos_idx, best_layer].item())

results_text = f"""Run: {timestamp}
Model: {MODEL_PATH}
Device: {DEVICE}
Training pairs: {N_TRAIN}
Validation pairs: {N_VAL}

Selected direction: pos={best_neg_pos}, layer={best_layer}
  Baseline refusal score: {baseline_score:.4f}
  Ablation refusal score: {ablation_score:.4f}
  Change: {ablation_score - baseline_score:.4f}
"""

with open("results.txt", "w") as f:
    f.write(results_text)

print(f"\nResults saved to results.txt")
