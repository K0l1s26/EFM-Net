import json
import os
import matplotlib.pyplot as plt
from config.config import parse_args

cfg = parse_args()
def plot_history(json_path: str, save_dir: str | None = None):
    with open(json_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    val_idx = [i for i, v in enumerate(history["val_mean"], 1) if v is not None]
    val_mean = [v for v in history["val_mean"] if v is not None]
    val_tc = [v for v in history["val_tc"] if v is not None]
    val_wt = [v for v in history["val_wt"] if v is not None]
    val_et = [v for v in history["val_et"] if v is not None]

    fig1, ax1l = plt.subplots(figsize=(10, 5))
    ax1r = ax1l.twinx()

    ax1l.plot(epochs, history["train_loss"], color="#2c3e50", linewidth=1.5, label="Train Loss")
    ax1r.plot(val_idx, val_mean, color="#e74c3c", linewidth=1.8,
              marker="o", markersize=4, label="Val Mean Dice")

    ax1l.set_xlabel("Epoch")
    ax1l.set_ylabel("Loss", color="#2c3e50")
    ax1r.set_ylabel("Dice", color="#e74c3c")
    ax1l.tick_params(axis="y", labelcolor="#2c3e50")
    ax1r.tick_params(axis="y", labelcolor="#e74c3c")

    lines1, labels1 = ax1l.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1l.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1l.set_title("Train Loss & Val Mean Dice")
    ax1l.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    for label, values, color in [
        ("TC (Tumor Core)", val_tc, "#e74c3c"),
        ("WT (Whole Tumor)", val_wt, "#2ecc71"),
        ("ET (Enhancing Tumor)", val_et, "#3498db"),
    ]:
        ax2.plot(val_idx, values, linewidth=1.8, marker="o", markersize=4,
                 label=label, color=color)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice")
    ax2.set_title("Validation Dice — WT / TC / ET")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path1 = os.path.join(save_dir, "loss_and_mean_dice.png")
        path2 = os.path.join(save_dir, "wt_tc_et_dice.png")
        fig1.savefig(path1, dpi=150, bbox_inches="tight")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        print(f"📈 The curve graph has been saved:\n  {path1}\n  {path2}")
    else:
        plt.show()

    plt.close("all")


