# Refusal Direction Analysis - Gemma-2B-IT

Analyzing and ablating refusal mechanisms in the Gemma-2B-IT model using harmful vs. harmless instruction pairs.

Brief Methodology:
1. Downloads harmful instructions from LLM-attacks dataset and harmless instructions from Alpaca
2. Extracts activation differences between harmful and harmless prompts
3. Tests which layer/position combination, when ablated, minimizes the refusal score
4. Reports the best direction found

Results.txt contains latest run's findings

### Key Findings

- Best ablation target: Layer 10, position -1 (chosen because last token before model response)
- Refusal Score Change: 5.8301 → -13.8001 (drop of 19.63 points; used log odds)
- Layer 10 Effect: Ablating the refusal direction at layer 10 switches model from strong refusal to strong compliance; Layer 10 out of ~18 total layers had the largest effect which makes sense because the layer is mid-to-late in the network where refusal signals are integrated

### To Do

Melissa is replicating on other VLMs, but this code can also easily be reworked for that task if necessary.
