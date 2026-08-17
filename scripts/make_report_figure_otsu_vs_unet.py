"""One-off script: build the Otsu-vs-U-Net qualitative comparison figure used in the
report (Task 3/Q2). Reuses src.otsu_vs_unet output to pick the two example images
(closest U-Net/Otsu margin and largest margin) and renders input / GT / Otsu / U-Net
panels for each.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src import config, data_prep
from src.classical_features import segment_otsu
from src.unet import UNet

with open(config.JSON_DIR / "otsu_vs_unet.json") as f:
    summary = json.load(f)

ids = [summary["example_unet_wins"]["image_id"], summary["example_otsu_wins"]["image_id"]]
labels = ["largest U-Net margin", "smallest U-Net margin"]

device = "cpu"
model = UNet().to(device)
model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "unet_best.pt", map_location=device))
model.eval()

fig, axes = plt.subplots(2, 4, figsize=(11, 6))
for row, (image_id, label) in enumerate(zip(ids, labels)):
    gray = data_prep.load_and_preprocess("val", image_id)
    gt = np.array(Image.open(config.DATA_DIR / "val" / "masks" / f"{image_id}.png").convert("L")) > 127
    otsu_mask = segment_otsu(gray)
    with torch.no_grad():
        t = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None]
        unet_mask = (torch.sigmoid(model(t)) > 0.5)[0, 0].numpy()

    panels = [gray, gt, otsu_mask, unet_mask]
    titles = [f"{image_id} input", "ground truth", "Otsu", "U-Net"]
    for col, (panel, title) in enumerate(zip(panels, titles)):
        ax = axes[row, col]
        ax.imshow(panel, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    axes[row, 0].set_ylabel(label, fontsize=9)

r = next(r for r in summary["per_image"] if r["image_id"] == ids[0])
r2 = next(r for r in summary["per_image"] if r["image_id"] == ids[1])
fig.suptitle(
    f"Otsu vs U-Net -- {ids[0]}: Otsu Dice={r['otsu_dice']:.3f}, U-Net Dice={r['unet_dice']:.3f}   |   "
    f"{ids[1]}: Otsu Dice={r2['otsu_dice']:.3f}, U-Net Dice={r2['unet_dice']:.3f}",
    fontsize=10,
)
fig.tight_layout()
out_path = config.FIGURES_DIR / "otsu_vs_unet_panels.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
