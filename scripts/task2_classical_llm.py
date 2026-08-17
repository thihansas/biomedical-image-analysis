"""Task 2: Otsu + regionprops classical features, then numbers-first LLM interpretation.

Run from the project root:
    python scripts/task2_classical_llm.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import classical_features  # noqa: E402


def main():
    print("=== Task 2: classical segmentation + regionprops + numbers-first LLM ===")
    record = classical_features.run_task2()
    print("\nNumeric summary sent to the LLM:\n", record["numeric_summary"])
    print("\nLLM paragraph:\n", record["llm_paragraph"])
    print("\nLLM JSON:\n", record["llm_json"])
    print("\nCode-derived JSON (source of truth):\n", record["code_derived_json"])
    print("\nSaved full record to outputs/json_records/task2_classical_llm.json")


if __name__ == "__main__":
    main()
