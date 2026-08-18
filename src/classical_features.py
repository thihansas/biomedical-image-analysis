
import json

import numpy as np
import pandas as pd
from skimage import measure, morphology
from skimage.filters import threshold_otsu

from . import config, data_prep, llm_utils

REGIONPROPS_FEATURES = (
    "label",
    "area",
    "eccentricity",
    "solidity",
    "mean_intensity",
    "perimeter",
    "major_axis_length",
    "minor_axis_length",
    "extent",
)

INTERPRETATION_PROMPT_TEMPLATE = """You are an assistant that interprets a table of \
image-analysis measurements for a biomedical research image. You are NOT given the \
image itself, only these numeric measurements, and you must not invent visual \
details that are not implied by the numbers. You are NOT a diagnostic tool: do not \
name a disease or suggest treatment.

Measurements summary (from classical segmentation of one image):
{summary_text}

Using only these numbers:
1. Write ONE short paragraph (3-4 sentences) describing what the object population \
looks like (count, density, size spread, shape regularity).
2. Then output a JSON object, on its own, with exactly this schema:
{{"n_objects": <int>, "density_class": "sparse|normal|dense", "shape_regularity": \
"regular|irregular", "quality_flag": "ok|uncertain"}}

If the numbers are ambiguous or too few objects were detected to judge shape, use \
"uncertain" for quality_flag rather than guessing.

Give the paragraph first, then the JSON object, no markdown fences.
"""


def segment_otsu(gray_img: np.ndarray, min_object_size: int = 15) -> np.ndarray:
    """Otsu threshold + morphological cleanup -> boolean foreground mask.

    Nuclei are bright-on-dark, so the foreground is pixels *above* the Otsu
    threshold. Small opening removes speckle noise; removing small objects
    discards leftover single-pixel/few-pixel noise blobs.
    """
    thresh = threshold_otsu(gray_img)
    binary = gray_img > thresh
    binary = morphology.opening(binary, morphology.disk(1))
    binary = morphology.remove_small_objects(binary, min_size=min_object_size)
    binary = morphology.closing(binary, morphology.disk(2))
    return binary


def extract_region_table(gray_img: np.ndarray, binary_mask: np.ndarray) -> pd.DataFrame:
    """Label connected components and compute a regionprops feature table."""
    labels = measure.label(binary_mask)
    if labels.max() == 0:
        return pd.DataFrame(columns=REGIONPROPS_FEATURES)
    table = measure.regionprops_table(labels, intensity_image=gray_img, properties=REGIONPROPS_FEATURES)
    return pd.DataFrame(table)


def summarise_table(df: pd.DataFrame) -> str:
    """Turn a regionprops DataFrame into a short natural-language numeric summary."""
    if df.empty:
        return "No objects were detected above the segmentation threshold."

    n = len(df)
    lines = [
        f"n_objects = {n}",
        f"area: mean={df['area'].mean():.1f}, std={df['area'].std():.1f}, "
        f"min={df['area'].min():.1f}, max={df['area'].max():.1f} (pixels)",
        f"eccentricity: mean={df['eccentricity'].mean():.3f} (0=circle, close to 1=elongated)",
        f"solidity: mean={df['solidity'].mean():.3f} (fraction of convex hull filled; "
        f"lower means more irregular/concave shape)",
        f"mean_intensity: mean={df['mean_intensity'].mean():.1f} (per-object mean pixel value)",
        f"extent: mean={df['extent'].mean():.3f} (fraction of bounding box filled)",
    ]
    return "\n".join(lines)


def derive_heuristic_json(df: pd.DataFrame) -> dict:
    """A deterministic, code-computed version of the same JSON schema we ask the LLM for.

    This is the 'source of truth' companion to the LLM's JSON: if the LLM's
    JSON disagrees with this, the code-derived version is what should be
    trusted downstream (see report Q4 on hallucination mitigation).
    """
    n = len(df)
    if n == 0:
        return {"n_objects": 0, "density_class": "sparse", "shape_regularity": "uncertain", "quality_flag": "uncertain"}

    if n < 15:
        density_class = "sparse"
    elif n <= 40:
        density_class = "normal"
    else:
        density_class = "dense"

    shape_regularity = "regular" if (df["solidity"].mean() > 0.9 and df["eccentricity"].mean() < 0.6) else "irregular"
    quality_flag = "ok" if n >= 3 else "uncertain"

    return {
        "n_objects": int(n),
        "density_class": density_class,
        "shape_regularity": shape_regularity,
        "quality_flag": quality_flag,
        "mean_area": float(df["area"].mean()),
    }


def interpret_with_llm(summary_text: str) -> tuple[str, dict]:
    """Send the numeric summary (no image) to the local text LLM; return (paragraph, json)."""
    prompt = INTERPRETATION_PROMPT_TEMPLATE.format(summary_text=summary_text)
    parsed_json, raw_text = llm_utils.chat_json(config.TEXT_LLM_MODEL, prompt)
    # Paragraph is whatever text precedes the JSON block in the raw response.
    json_start = raw_text.find("{")
    paragraph = raw_text[:json_start].strip() if json_start > 0 else raw_text.strip()
    return paragraph, parsed_json


def run_task2(split: str = "train", image_id: str = "train_004") -> dict:
    """Full Task 2 pipeline on one representative image; saves + returns the record."""
    gray_img = data_prep.load_and_preprocess(split, image_id)
    binary_mask = segment_otsu(gray_img)
    df = extract_region_table(gray_img, binary_mask)
    summary_text = summarise_table(df)
    paragraph, llm_json = interpret_with_llm(summary_text)
    code_json = derive_heuristic_json(df)

    record = {
        "image_id": image_id,
        "split": split,
        "n_objects_detected": int(len(df)),
        "region_table_head": df.head(10).to_dict(orient="records"),
        "numeric_summary": summary_text,
        "llm_paragraph": paragraph,
        "llm_json": llm_json,
        "code_derived_json": code_json,
    }

    out_path = config.JSON_DIR / "task2_classical_llm.json"
    out_path.write_text(json.dumps(record, indent=2, default=str))
    df.to_csv(config.JSON_DIR / f"task2_region_table_{image_id}.csv", index=False)
    return record


if __name__ == "__main__":
    result = run_task2()
    print(json.dumps({k: v for k, v in result.items() if k != "region_table_head"}, indent=2, default=str))
