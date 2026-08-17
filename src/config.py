"""Central configuration: paths, image size, split names, and the Ollama models used.

Kept in one place so every script/notebook agrees on where data lives and which
local models to call, without hard-coding paths throughout the codebase.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "nuclei_dataset"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
JSON_DIR = OUTPUT_DIR / "json_records"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for d in (OUTPUT_DIR, FIGURES_DIR, JSON_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "val", "test")
IMG_SIZE = 256  # dataset already ships at 256x256; kept explicit for the resize step

# --- Ollama models -------------------------------------------------------
# Vision-capable model used only for the direct image description (Task 1).
VLM_MODEL = "llama3.2-vision"
# Lightweight text-only model used for numbers-only interpretation (Task 2, Task 4).
# It never receives image data, only the regionprops summary text.
TEXT_LLM_MODEL = "llama3.2"

RANDOM_SEED = 42
