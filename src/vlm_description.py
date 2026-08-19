
import json
from pathlib import Path

from . import config, llm_utils

NAIVE_PROMPT = (
    "What is this image showing? Please describe it and tell me if there's "
    "anything wrong with the tissue."
)

OPTIMISED_PROMPT = """You are an assistant that provides objective, descriptive \
observations about a biomedical research image, for educational purposes only. \
You are NOT a diagnostic tool: you must not name a disease, give a diagnosis, or \
suggest treatment. Describe only what is visually observable.

Look at the image and report:
- modality: the imaging/acquisition type you can visually infer (e.g. fluorescence \
microscopy, brightfield, MRI, X-ray). If you cannot tell, write "uncertain".
- tissue_type: the apparent tissue or specimen type. If you cannot tell, write \
"uncertain".
- notable_features: 1-2 sentences of purely descriptive observations (shape, \
distribution, staining/contrast pattern of the objects you see). Do not speculate \
about pathology.
- image_quality: a brief comment on focus, contrast, and noise/artifacts. If you \
cannot judge it, write "uncertain".

If you are not confident about any field, write "uncertain" for that field instead \
of guessing.

Respond with ONLY a single JSON object, no extra commentary, no markdown fences, in \
exactly this schema:
{"modality": "...", "tissue_type": "...", "notable_features": "...", "image_quality": "..."}
"""


def _image_path(split: str, image_id: str) -> str:
    return str(config.DATA_DIR / split / "images" / f"{image_id}.png")


def describe_naive(split: str, image_id: str) -> str:
    """Free-text response to the naive prompt, no JSON schema requested."""
    img_path = _image_path(split, image_id)
    return llm_utils.chat_text(config.VLM_MODEL, NAIVE_PROMPT, images=[img_path])


def describe_optimised(split: str, image_id: str) -> tuple[dict, str]:
    """Structured JSON response to the optimised, descriptive-not-diagnostic prompt."""
    img_path = _image_path(split, image_id)
    return llm_utils.chat_json(config.VLM_MODEL, OPTIMISED_PROMPT, images=[img_path])


def repeated_runs(split: str, image_id: str, n: int = 3) -> list[dict]:
    """Run the optimised prompt n times on the same image, to show run-to-run variability."""
    img_path = _image_path(split, image_id)
    runs = []
    for i in range(n):
        try:
            parsed, raw = llm_utils.chat_json(config.VLM_MODEL, OPTIMISED_PROMPT, images=[img_path])
            runs.append({"run": i + 1, "json": parsed, "raw": raw})
        except RuntimeError as exc:
            runs.append({"run": i + 1, "json": None, "raw": str(exc)})
    return runs


def run_task1_vlm(split: str = "train", image_id: str = "train_004", n_repeats: int = 3) -> dict:
    """Full Task 1 VLM comparison (naive vs optimised, repeated-run variability), saved + returned."""
    naive_text = describe_naive(split, image_id)
    optimised_json, optimised_raw = describe_optimised(split, image_id)
    variability = repeated_runs(split, image_id, n=n_repeats)

    record = {
        "image_id": image_id,
        "split": split,
        "naive_prompt": NAIVE_PROMPT,
        "naive_response": naive_text,
        "optimised_prompt": OPTIMISED_PROMPT,
        "optimised_response_json": optimised_json,
        "optimised_response_raw": optimised_raw,
        "repeated_runs": variability,
    }

    out_path = config.JSON_DIR / "task1_vlm_description.json"
    out_path.write_text(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    result = run_task1_vlm()
    print(json.dumps(result, indent=2)[:2000])
