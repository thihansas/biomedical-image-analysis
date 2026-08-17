"""Task 4: the full hybrid pipeline on unseen test images.

raw image -> U-Net mask -> regionprops feature table -> LLM structured JSON
record -> one-paragraph narrative. Records are aggregated into a single CSV.

The JSON record's numeric fields (n_objects, mean_area, density_class) are
always taken from the code-computed regionprops table, never from the LLM's
own arithmetic -- the LLM only supplies the narrative paragraph and a
free-text quality_flag. This is what keeps the structured record the
"source of truth": even if the LLM narrative drifts, the numbers it is
narrating are fixed and auditable. See classical_features.derive_heuristic_json
for the equivalent used in Task 2.
"""
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import config, data_prep, llm_utils
from .classical_features import extract_region_table, summarise_table
from .unet import UNet

NARRATIVE_PROMPT_TEMPLATE = """You are an assistant that writes a short, objective \
narrative for a biomedical research image, for educational purposes only. You are \
NOT a diagnostic tool: do not name a disease or suggest treatment. You are given \
only numeric measurements below, not the image itself -- do not invent visual \
details beyond what the numbers imply.

Image id: {image_id}
Measurements from automatic segmentation:
{summary_text}

Write ONE short paragraph (3-4 sentences) describing the object population implied \
by these numbers (count, density, size spread, shape regularity). If the numbers \
are ambiguous, say so rather than guessing. Return plain text only, no JSON, no \
markdown.
"""


def _density_class(n_objects: int) -> str:
    if n_objects < 15:
        return "sparse"
    if n_objects <= 40:
        return "normal"
    return "dense"


@torch.no_grad()
def run_unet_inference(model: UNet, gray_img: np.ndarray, device: str) -> np.ndarray:
    img_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)[None, None].to(device)
    return (torch.sigmoid(model(img_tensor)) > 0.5)[0, 0].cpu().numpy()


def process_image(model: UNet, device: str, split: str, image_id: str) -> dict:
    """Run one test image through the full hybrid pipeline; returns its JSON record + narrative."""
    gray_img = data_prep.load_and_preprocess(split, image_id)
    pred_mask = run_unet_inference(model, gray_img, device)
    df = extract_region_table(gray_img, pred_mask)
    summary_text = summarise_table(df)

    n_objects = int(len(df))
    mean_area = float(df["area"].mean()) if n_objects else 0.0
    density_class = _density_class(n_objects)
    quality_flag = "ok" if n_objects >= 3 else "uncertain"

    narrative = llm_utils.chat_text(
        config.TEXT_LLM_MODEL,
        NARRATIVE_PROMPT_TEMPLATE.format(image_id=image_id, summary_text=summary_text),
    )

    record = {
        "image_id": image_id,
        "n_objects": n_objects,
        "mean_area": mean_area,
        "density_class": density_class,
        "quality_flag": quality_flag,
        "narrative": narrative,
    }
    return record


def run_task4(split: str = "test", checkpoint_path=None) -> pd.DataFrame:
    """Run the hybrid pipeline over every image in `split`, save per-image JSON
    records and one aggregated CSV. Returns the aggregated DataFrame."""
    checkpoint_path = checkpoint_path or config.CHECKPOINT_DIR / "unet_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    records = []
    for image_id in data_prep.list_image_ids(split):
        record = process_image(model, device, split, image_id)
        records.append(record)
        print(f"{image_id}: n_objects={record['n_objects']} density={record['density_class']}")

    (config.JSON_DIR / "task4_hybrid_records.json").write_text(json.dumps(records, indent=2))

    df = pd.DataFrame(records)
    csv_path = config.OUTPUT_DIR / "task4_hybrid_records.csv"
    df.to_csv(csv_path, index=False)
    return df


if __name__ == "__main__":
    df = run_task4("test")
    print(df.to_string(index=False))
