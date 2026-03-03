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

## To Do

Melissa is replicating on other VLMs, but this code can also easily be reworked for that task if necessary.
