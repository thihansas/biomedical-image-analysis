
import matplotlib.pyplot as plt
import numpy as np
import torch

from src import config, data_prep
from src.hybrid_pipeline import run_unet_inference
from src.robustness import _load_corrupted
from src.unet import UNet

device = "cpu"
model = UNet().to(device)
model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "unet_best.pt", map_location=device))
model.eval()

base_id = "test_000"
clean = data_prep.load_and_preprocess("test", base_id)
blur = _load_corrupted(f"{base_id}_blur.png")
lowc = _load_corrupted(f"{base_id}_lowcontrast.png")

variants = [("clean", clean), ("heavy blur", blur), ("low contrast", lowc)]

def _bordered(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)


fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2))
for col, (name, img) in enumerate(variants):
    pred = run_unet_inference(model, img, device)
    axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(f"{name}\ninput", fontsize=10)
    _bordered(axes[0, col])
    axes[1, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
    n_fg_pct = 100 * pred.mean()
    axes[1, col].set_title(f"U-Net mask ({n_fg_pct:.0f}% fg)", fontsize=10)
    _bordered(axes[1, col])

fig.suptitle(f"Corruption propagation ({base_id}): input (top) and predicted mask (bottom)", fontsize=11)
fig.tight_layout()
out_path = config.FIGURES_DIR / "robustness_panels.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
