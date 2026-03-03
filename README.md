# Refusal Direction Analysis - Gemma-2B-IT

Analyzing and ablating refusal mechanisms in the Gemma-2B-IT model using harmful vs. harmless instruction pairs.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## How it works

1. Downloads harmful instructions from LLM-attacks dataset and harmless instructions from Alpaca
2. Extracts activation differences between harmful and harmless prompts
3. Tests which layer/position combination, when ablated, minimizes the refusal score
4. Reports the best direction found

## Results

See `results.txt` for the latest run results.

### Key Findings

- **Best ablation target**: Layer 10, position -1 (last token before model response)
- **Refusal score change**: 5.8301 → -13.8001 (drop of 19.63 points)
- **Effect**: Ablating the refusal direction at layer 10 switches model from strong refusal to strong compliance
- **Position consistency**: Most effective at the token position immediately before the model begins generating its response
- **Layer specificity**: Layer 10 out of ~18 total layers - mid-to-late in the network where refusal signals are integrated

## To Do

Melissa is replicating on other VLMs, but this code can also easily be reworked for that task if necessary.
