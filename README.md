# Cross-Modal Refusal Vector Transfer

Polishing of the original report was assisted by Claude.

## Running on GCP VM

```bash
tmux new -s experiment
pip install -r requirements.txt
python -c "from huggingface_hub import login; login()"
python experiment-2.py
```

If tab closes:
```bash
tmux attach -t experiment
```
