"""Task 1: preprocessing + EDA, then naive-vs-optimised VLM description with llama3.2-vision.

Run from the project root:
    python scripts/task1_data_and_vlm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_prep, vlm_description  # noqa: E402


def main():
    print("=== Task 1a: preprocessing + EDA (train split) ===")
    eda_summary = data_prep.run_eda("train")
    print(eda_summary)

    print("\n=== Task 1b: VLM description (llama3.2-vision) ===")
    print("This calls llama3.2-vision 5 times (naive + optimised + 3 repeats) "
          "and can take several minutes on CPU.")
    vlm_record = vlm_description.run_task1_vlm()
    print("\nNaive response:\n", vlm_record["naive_response"])
    print("\nOptimised JSON:\n", vlm_record["optimised_response_json"])
    print(f"\nSaved full record to outputs/json_records/task1_vlm_description.json")


if __name__ == "__main__":
    main()
