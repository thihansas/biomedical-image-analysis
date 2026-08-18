
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
# If this fails to load with "unknown model architecture: 'mllama'", that is an
# upstream Ollama bug on newer Ollama versions.
# Pin Ollama to v0.23.4, or swap in an alternative vision model here 
# (qwen2.5vl / qwen3-vl / a ministral vision variant are
# acceptable substitutes) -- everything downstream only ever reads config.VLM_MODEL.
VLM_MODEL = "llama3.2-vision"
# Lightweight text-only model used for numbers-only interpretation (Task 2, Task 4).
# It never receives image data, only the regionprops summary text.
TEXT_LLM_MODEL = "llama3.2"

RANDOM_SEED = 42
