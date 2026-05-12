import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TEXT_JSON = os.path.join(RESULTS_DIR, "text_refusal_vector_layer_analysis.json")
IMAGE_JSON = os.path.join(RESULTS_DIR, "image_refusal_vector_layer_analysis.json")


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_layer_cosine(data, title, subtitle, out_path):
    cos_sims = data["cos_sims"]
    layers = sorted(int(k) for k in cos_sims)
    sims = [cos_sims[str(l)] for l in layers]
    threshold = data["threshold"]
    start = data["refusal_layer_start"]
    end = data["refusal_layer_end"]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Shade the selected refusal window
    ax.axvspan(start - 0.5, end + 0.5, color="#4a90d9", alpha=0.12, label=f"Refusal window (layers {start}–{end})")

    # Threshold line
    ax.axhline(threshold, color="#e05c5c", linewidth=1.4, linestyle="--", label=f"Refusal threshold = {threshold}")

    # Zero line
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle=":")

    # Color points by above/below threshold
    colors = ["#4a90d9" if s >= threshold else "#aaaaaa" for s in sims]
    ax.plot(layers, sims, color="#cccccc", linewidth=1.2, zorder=1)
    ax.scatter(layers, sims, c=colors, s=48, zorder=2)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_ylim(-1.0, 1.05)
    ax.set_xlim(-0.5, max(layers) + 0.5)

    above_patch = mpatches.Patch(color="#4a90d9", label=f"cos_sim ≥ {threshold}")
    below_patch = mpatches.Patch(color="#aaaaaa", label=f"cos_sim < {threshold}")
    window_patch = mpatches.Patch(color="#4a90d9", alpha=0.25, label=f"Refusal window (layers {start}–{end})")
    threshold_line = plt.Line2D([0], [0], color="#e05c5c", linewidth=1.4, linestyle="--", label=f"Refusal threshold = {threshold}")
    ax.legend(handles=[threshold_line, window_patch, above_patch, below_patch], fontsize=10, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    text_data = load(TEXT_JSON)
    image_data = load(IMAGE_JSON)

    plot_layer_cosine(
        text_data,
        title="Text-Derived Refusal Vector — Layer Cosine Similarities",
        subtitle="Cosine similarity between the image refusal direction and mean refused-text activations per layer",
        out_path=os.path.join(FIGURES_DIR, "text_refusal_vector_layers.png"),
    )

    plot_layer_cosine(
        image_data,
        title="Image-Derived Refusal Vector — Layer Cosine Similarities",
        subtitle="Cosine similarity between the text refusal direction and mean refused-image activations per layer",
        out_path=os.path.join(FIGURES_DIR, "image_refusal_vector_layers.png"),
    )
