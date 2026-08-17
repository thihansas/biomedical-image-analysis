"""Compare classical Otsu segmentation against the trained U-Net on the same split.

Produces per-image Dice/IoU for both methods (against the same ground-truth
masks) so the report can answer "did the U-Net improve on classical Otsu?"
with numbers and a concrete example image for each side.
"""
import json

import numpy as np
import torch
from PIL import Image

from . import config, data_prep
from .classical_features import segment_otsu
from .metrics import dice_score, iou_score
from .unet import UNet


@torch.no_grad()
def compare_on_split(split: str = "val", checkpoint_path=None) -> dict:
    checkpoint_path = checkpoint_path or config.CHECKPOINT_DIR / "unet_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    rows = []
    for image_id in data_prep.list_image_ids(split):
        gray_img = data_prep.load_and_preprocess(split, image_id)
        gt_mask = np.array(Image.open(config.DATA_DIR / split / "masks" / f"{image_id}.png").convert("L")) > 127

        otsu_mask = segment_otsu(gray_img)

        img_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)[None, None].to(device)
        unet_mask = (torch.sigmoid(model(img_tensor)) > 0.5)[0, 0].cpu().numpy()

        rows.append({
            "image_id": image_id,
            "otsu_dice": dice_score(otsu_mask, gt_mask),
            "otsu_iou": iou_score(otsu_mask, gt_mask),
            "unet_dice": dice_score(unet_mask, gt_mask),
            "unet_iou": iou_score(unet_mask, gt_mask),
        })

    mean_otsu_dice = float(np.mean([r["otsu_dice"] for r in rows]))
    mean_unet_dice = float(np.mean([r["unet_dice"] for r in rows]))

    best_for_unet = max(rows, key=lambda r: r["unet_dice"] - r["otsu_dice"])
    best_for_otsu = max(rows, key=lambda r: r["otsu_dice"] - r["unet_dice"])

    summary = {
        "split": split,
        "mean_otsu_dice": mean_otsu_dice,
        "mean_unet_dice": mean_unet_dice,
        "mean_otsu_iou": float(np.mean([r["otsu_iou"] for r in rows])),
        "mean_unet_iou": float(np.mean([r["unet_iou"] for r in rows])),
        "example_unet_wins": best_for_unet,
        "example_otsu_wins": best_for_otsu,
        "per_image": rows,
    }

    (config.JSON_DIR / "otsu_vs_unet.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    summary = compare_on_split("val")
    print(f"mean Otsu Dice = {summary['mean_otsu_dice']:.4f}")
    print(f"mean U-Net Dice = {summary['mean_unet_dice']:.4f}")
    print("U-Net wins most on:", summary["example_unet_wins"]["image_id"])
    print("Otsu wins most on:", summary["example_otsu_wins"]["image_id"])
