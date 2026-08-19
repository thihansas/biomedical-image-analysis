# AI Imaging Case Study: Fluorescence-Microscopy Nuclei Pipeline

A hybrid biomedical image-analysis pipeline: raw image, then U-Net or classical
segmentation, then quantitative region features, then a structured JSON record
and narrative, all evaluated against a local multimodal VLM description. Every
LLM call runs locally through [Ollama](https://ollama.com); there are no cloud APIs
involved.

**Modality / dataset:** synthetic fluorescence-microscopy nuclei images (DAPI-like
staining), from https://github.com/Nickolay-K/Assingnment-3-dataset. 256x256 RGB
images with paired binary masks and instance labels, split train(80)/val(20)/test(12),
plus 4 corrupted test-image variants (heavy blur / low contrast) for the robustness
extension. `data/nuclei_dataset/README.md` documents the dataset itself.

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
  report.pdf                rendered report, THE FILE TO SUBMIT (exactly 4 pages)
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
   `llama3.2-vision` (11B, Q4_K_M) needs roughly **11GB of free RAM** to load. If it
   fails with a "requires more system memory" error, close other memory-heavy
   applications and try again.

   One thing to watch for: Ollama's newer inference engine (v0.30.0 onward, as of
   testing in August 2026) fails to load `llama3.2-vision`, throwing
   `error loading model: unknown model architecture: 'mllama'`
   ([ollama/ollama#16490](https://github.com/ollama/ollama/issues/16490),
   [#16547](https://github.com/ollama/ollama/issues/16547)). That's an upstream Ollama
   bug rather than anything in this code. I developed and ran this against
   **Ollama v0.23.4**, the last release before the regression, which loads
   `llama3.2-vision` fine. If Task 1's VLM step hits the `mllama` error on your
   machine, grab v0.23.4 from
   https://github.com/ollama/ollama/releases/tag/v0.23.4 (Windows: `OllamaSetup.exe`
   on that page) instead of the current release.

   The lecturer has confirmed this fallback for anyone hitting the same bug: swap
   in a different local vision model if downgrading Ollama isn't practical (say the
   install is managed and can't be rolled back) — `qwen2.5vl`, `qwen3-vl`, or a
   `ministral` vision variant all work, as does running `llama3.2-vision` in Colab
   (see the Lab 2 notebook) rather than locally. To switch models, just change
   `VLM_MODEL` in `src/config.py` to the pulled model's tag (e.g.
   `ollama pull qwen2.5vl` then `VLM_MODEL = "qwen2.5vl"`); nothing else needs to
   change since `src/vlm_description.py` and the rest of the pipeline only ever read
   `config.VLM_MODEL`. Keep in mind that a different model will give different VLM
   outputs than what's already saved in `outputs/json_records/task1_vlm_description.json`
   and quoted in the report — regenerate those before trusting the report text if you
   re-run Task 1 on a new model.
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

`report/report.html` is the report source and renders to exactly 4 pages; `report/report.pdf`
is the version submitted alongside the code. After editing the HTML, regenerate the PDF with
a headless Chromium/Edge print, e.g.:
```
msedge --headless --disable-gpu --no-sandbox --user-data-dir=<scratch-dir> \
  --no-pdf-header-footer --print-to-pdf=report/report.pdf report/report.html
```

Every script prints a summary to the console and writes its figures/JSON/CSV to
`outputs/`. Figures are saved at the resolution used in the report, and the JSON
records keep the exact prompts used — so the "optimised prompt" text quoted in the
report is copied straight out of `outputs/json_records/*.json` and the `*_PROMPT`
constants in `src/vlm_description.py` and `src/classical_features.py`.

## Design notes

Otsu + morphology rather than deep learning for Task 2 keeps the numbers the LLM
sees fully deterministic and auditable — that determinism is the whole point of
comparing the numbers-first approach against the direct VLM description.

The LLM never computes the JSON's numeric fields itself in Task 2 or Task 4. Look at
`classical_features.derive_heuristic_json` and `hybrid_pipeline.process_image`:
`n_objects`, `mean_area`, and `density_class` all come from the regionprops table in
code, and the LLM only ever produces the free-text narrative/paragraph plus a
qualitative flag. A hallucinated LLM output therefore can't corrupt the structured
"source of truth" record.

The U-Net is deliberately small (base width 16, a few hundred thousand parameters),
since there are only 80 training images and training runs on CPU — a full-width
U-Net would just be slower to train and more likely to overfit at this scale.

Task 1 uses the vision model (`llama3.2-vision`) because it's given the actual
image. Tasks 2 and 4 use the smaller text-only model (`llama3.2`) on purpose,
since they only ever see numbers rather than pixels; that's cheaper and faster,
and it makes the numbers-only boundary explicit in the code rather than just in
the prompt wording.
