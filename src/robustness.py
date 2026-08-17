"""Extension: trace how image corruption propagates through the pipeline.

The dataset ships pre-corrupted variants of test_000 and test_004 (heavy
blur and low-contrast) in test_corrupted/images. For each corrupted image we
compare against its clean counterpart at four stages:
  1. raw pixel statistics (mean/std intensity)
  2. U-Net mask overlap (Dice/IoU) against the ORIGINAL ground-truth mask
  3. the regionprops feature table (n_objects, mean area)
  4. the LLM narrative/JSON record

This lets us identify the earliest stage where the corruption's effect
becomes large relative to the clean image's own numbers.
"""
import json

import numpy as np
import torch
from PIL import Image

from . import config, data_prep
from .classical_features import extract_region_table, summarise_table
from .hybrid_pipeline import run_unet_inference
from .metrics import dice_score, iou_score
from .unet import UNet

CORRUPTED_DIR = config.DATA_DIR / "test_corrupted" / "images"


def _load_corrupted(filename: str) -> np.ndarray:
    img = Image.open(CORRUPTED_DIR / filename).convert("L")
    if img.size != (config.IMG_SIZE, config.IMG_SIZE):
        img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def trace_one(model: UNet, device: str, base_image_id: str, corrupted_filename: str) -> dict:
    clean_img = data_prep.load_and_preprocess("test", base_image_id)
    corrupt_img = _load_corrupted(corrupted_filename)
    gt_mask = np.array(Image.open(config.DATA_DIR / "test" / "masks" / f"{base_image_id}.png").convert("L")) > 127

    # Stage 1: raw pixel stats
    stage1 = {
        "clean_mean": float(clean_img.mean()), "corrupt_mean": float(corrupt_img.mean()),
        "clean_std": float(clean_img.std()), "corrupt_std": float(corrupt_img.std()),
    }
    stage1["mean_pct_change"] = 100 * (stage1["corrupt_mean"] - stage1["clean_mean"]) / (stage1["clean_mean"] + 1e-7)
    stage1["std_pct_change"] = 100 * (stage1["corrupt_std"] - stage1["clean_std"]) / (stage1["clean_std"] + 1e-7)

    # Stage 2: U-Net mask overlap vs the ORIGINAL ground truth
    clean_pred = run_unet_inference(model, clean_img, device)
    corrupt_pred = run_unet_inference(model, corrupt_img, device)
    stage2 = {
        "clean_dice_vs_gt": dice_score(clean_pred, gt_mask),
        "corrupt_dice_vs_gt": dice_score(corrupt_pred, gt_mask),
        "clean_iou_vs_gt": iou_score(clean_pred, gt_mask),
        "corrupt_iou_vs_gt": iou_score(corrupt_pred, gt_mask),
    }
    stage2["dice_drop"] = stage2["clean_dice_vs_gt"] - stage2["corrupt_dice_vs_gt"]

    # Stage 3: regionprops feature table
    clean_df = extract_region_table(clean_img, clean_pred)
    corrupt_df = extract_region_table(corrupt_img, corrupt_pred)
    stage3 = {
        "clean_n_objects": int(len(clean_df)), "corrupt_n_objects": int(len(corrupt_df)),
        "clean_mean_area": float(clean_df["area"].mean()) if len(clean_df) else 0.0,
        "corrupt_mean_area": float(corrupt_df["area"].mean()) if len(corrupt_df) else 0.0,
    }
    stage3["n_objects_pct_change"] = 100 * (stage3["corrupt_n_objects"] - stage3["clean_n_objects"]) / max(stage3["clean_n_objects"], 1)

    # Stage 4: LLM narrative on the corrupted feature table
    from . import llm_utils
    from .hybrid_pipeline import NARRATIVE_PROMPT_TEMPLATE
    corrupt_summary = summarise_table(corrupt_df)
    narrative = llm_utils.chat_text(
        config.TEXT_LLM_MODEL,
        NARRATIVE_PROMPT_TEMPLATE.format(image_id=corrupted_filename, summary_text=corrupt_summary),
    )

    return {
        "base_image_id": base_image_id,
        "corrupted_file": corrupted_filename,
        "stage1_pixel_stats": stage1,
        "stage2_mask_overlap": stage2,
        "stage3_feature_table": stage3,
        "stage4_llm_narrative": narrative,
    }


def run_robustness_trace(checkpoint_path=None) -> list[dict]:
    checkpoint_path = checkpoint_path or config.CHECKPOINT_DIR / "unet_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    pairs = [
        ("test_000", "test_000_blur.png"),
        ("test_000", "test_000_lowcontrast.png"),
        ("test_004", "test_004_blur.png"),
        ("test_004", "test_004_lowcontrast.png"),
    ]
    results = [trace_one(model, device, base_id, fname) for base_id, fname in pairs]

    (config.JSON_DIR / "robustness_trace.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    results = run_robustness_trace()
    for r in results:
        print(f"{r['corrupted_file']}: mean_pct_change={r['stage1_pixel_stats']['mean_pct_change']:.1f}%  "
              f"dice_drop={r['stage2_mask_overlap']['dice_drop']:.3f}  "
              f"n_objects_pct_change={r['stage3_feature_table']['n_objects_pct_change']:.1f}%")
