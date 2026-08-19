
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "nuclei_dataset"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
JSON_DIR = OUTPUT_DIR / "json_records"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for d in (OUTPUT_DIR, FIGURES_DIR, JSON_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "val", "test")
IMG_SIZE = 256  # images already ship at 256x256, kept explicit for the resize step

# Vision model for Task 1's direct image description. If this errors with
# "unknown model architecture: 'mllama'" that's a known upstream Ollama bug on
# newer Ollama builds; pin Ollama to v0.23.4, or point VLM_MODEL at another
# vision model (qwen2.5vl, qwen3-vl, a ministral vision variant). Nothing else
# in the pipeline hardcodes a model name, so swapping this is enough.
VLM_MODEL = "llama3.2-vision"
# Text-only model for the numbers-only interpretation in Task 2/4 (never sees images).
TEXT_LLM_MODEL = "llama3.2"

RANDOM_SEED = 42
