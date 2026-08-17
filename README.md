# AI Imaging Case Study: Fluorescence-Microscopy Nuclei Pipeline

A hybrid biomedical image-analysis pipeline built for Assignment 3: raw image ->
U-Net / classical segmentation -> quantitative region features -> structured JSON
record -> narrative, evaluated against a local multimodal VLM description. All
LLM calls run locally via [Ollama](https://ollama.com); no cloud APIs are used.

**Modality / dataset:** synthetic fluorescence-microscopy nuclei images (DAPI-like
staining), from https://github.com/Nickolay-K/Assingnment-3-dataset. 256x256 RGB
images with paired binary masks and instance labels, split train(80)/val(20)/test(12),
plus 4 corrupted test-image variants (heavy blur / low contrast) used for the
robustness extension. See `data/nuclei_dataset/README.md` for the dataset's own
documentation.

## Repository layout

```
data/nuclei_dataset/     dataset (images, masks, labels, metadata.csv)
src/                     all pipeline logic, as importable functions
  config.py                paths, image size, which Ollama models to use
  metrics.py                Dice / IoU
  data_prep.py              Task 1: grayscale+resize preprocessing, EDA figures
  llm_utils.py               shared Ollama call + JSON-parsing/retry helper
  vlm_description.py        Task 1: naive vs optimised VLM prompt, repeated-run demo
  classical_features.py     Task 2: Otsu + morphology + regionprops + LLM interpretation
  unet.py                   Task 3: small U-Net model + BCE/Dice/BCE+Dice losses
  dataset.py                 PyTorch Dataset for image/mask pairs
  train_unet.py              Task 3: training loop, Dice/IoU eval, curve + panel figures
  otsu_vs_unet.py            Otsu vs U-Net comparison on the same split
  hybrid_pipeline.py         Task 4: full pipeline on the test split -> aggregated CSV
  robustness.py               extension: corruption trace through the pipeline stages
scripts/                 thin runnable entry points, one per task (see below)
report/
  report.html              4-page report source (2-column, print-ready)
  report.pdf                rendered report -- THE FILE TO SUBMIT (exactly 4 pages)
  report.docx                editable Word version (single-column; same content, ~8 pages)
  build_docx.py              regenerates report.docx from scratch (python-docx)
outputs/
  figures/                 all saved plots (EDA, loss/Dice curves, prediction panels)
  json_records/             per-task JSON records (prompts, LLM outputs, results)
  checkpoints/              trained U-Net weights (unet_best.pt)
  task4_hybrid_records.csv  aggregated Task 4 output
```

## Setup

1. **Python dependencies**: `pip install -r requirements.txt` (tested with Python 3.12).
2. **Ollama**: install from https://ollama.com, then pull the two local models used:
   ```
   ollama pull llama3.2-vision
   ollama pull llama3.2
   ```
   `llama3.2-vision` (11B, Q4_K_M) needs roughly **11GB of free RAM** to load. Close
   other memory-heavy applications first if the load fails with a "requires more
   system memory" error.

   **Important Ollama version note:** Ollama's newer inference engine (v0.30.0
   onward, as of testing in August 2026) has a regression where it fails to load
   `llama3.2-vision` with `error loading model: unknown model architecture: 'mllama'`
   ([ollama/ollama#16490](https://github.com/ollama/ollama/issues/16490),
   [#16547](https://github.com/ollama/ollama/issues/16547)). This is an upstream
   Ollama bug, not a bug in this code. This pipeline was developed and run against
   **Ollama v0.23.4**, the last release before that regression, which loads
   `llama3.2-vision` correctly. If Task 1's VLM step fails with the `mllama`
   error on your machine, install v0.23.4 from
   https://github.com/ollama/ollama/releases/tag/v0.23.4 (Windows: `OllamaSetup.exe`
   from that release page) rather than the latest version.
3. The dataset is already included under `data/nuclei_dataset/`.

## Running

Each task is a standalone script, runnable from the project root:

```
python scripts/task1_data_and_vlm.py        # preprocessing, EDA, VLM description (~5-10 min, CPU)
python scripts/task2_classical_llm.py       # Otsu + regionprops + numbers-first LLM
python scripts/task3_train_unet.py          # trains U-Net (40 epochs, CPU, ~a few minutes), Otsu-vs-U-Net comparison
python scripts/task4_hybrid_pipeline.py     # full pipeline on test split -> CSV (needs Task 3's checkpoint)
python scripts/extension_robustness.py      # robustness extension (needs Task 3's checkpoint)
```

or run everything in sequence with `python scripts/run_all.py`.

Two extra one-off scripts regenerate the two report-only comparison figures (both need
Task 3's checkpoint and, for the second, Task 4's corrupted-image outputs):
```
python scripts/make_report_figure_otsu_vs_unet.py   # outputs/figures/otsu_vs_unet_panels.png
python scripts/make_report_figure_robustness.py     # outputs/figures/robustness_panels.png
```

`report/report.html` is the report source (renders to exactly 4 pages); `report/report.pdf`
is the rendered version submitted alongside the code. Regenerate the PDF after editing the
HTML with a headless Chromium/Edge print, e.g.:
```
msedge --headless --disable-gpu --no-sandbox --user-data-dir=<scratch-dir> \
  --no-pdf-header-footer --print-to-pdf=report/report.pdf report/report.html
```

Every script prints a summary to the console and writes its figures/JSON/CSV to
`outputs/`. Figures are saved at the resolution used in the report; JSON records
include the exact prompts used, so the "optimised prompt" text in the report is
copied directly from `outputs/json_records/*.json` / the `*_PROMPT` constants in
`src/vlm_description.py` and `src/classical_features.py`.

## Design notes

- **Otsu + morphology, not deep learning, for Task 2**: this keeps the numbers
  the LLM sees fully deterministic and auditable, which is the point of the
  numbers-first comparison against the direct VLM description.
- **The LLM never computes the JSON's numeric fields itself** in Task 2/4 -- see
  `classical_features.derive_heuristic_json` and `hybrid_pipeline.process_image`,
  where `n_objects`, `mean_area`, and `density_class` come from the regionprops
  table in code. The LLM only produces the free-text narrative/paragraph and a
  qualitative flag, so a hallucinated LLM output cannot corrupt the structured
  "source of truth" record.
- **U-Net is intentionally small** (base width 16, ~a few hundred thousand
  parameters) given only 80 training images and CPU-only training; a full-width
  U-Net would be slower to train and more prone to overfitting at this scale.
- **Text LLM (`llama3.2`) vs vision LLM (`llama3.2-vision`)**: Task 1 uses the
  vision model because it is given the actual image. Tasks 2 and 4 deliberately
  use the smaller text-only model because they are only ever given numbers, not
  pixels -- this is also cheaper/faster and makes the "numbers-only" boundary
  explicit in the code, not just in the prompt.
