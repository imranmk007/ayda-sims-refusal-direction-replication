# Cross-Modal Refusal Vector Transfer

## Running on GCP VM

```bash
tmux new -s experiment - optional but useful for GCP at least
sudo apt install pip
pip install -r requirements.txt
python3 -c "from huggingface_hub import login; login()"
Insert HF Token: In slack, titled "Imran HF Token: xx"
python3 <<filename.py>>
```

Relaunch tab:

```bash
tmux attach -t experiment
```
