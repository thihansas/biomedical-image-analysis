"""Run the entire pipeline (Tasks 1-4 + robustness extension) end to end.

Run from the project root:
    python scripts/run_all.py

Takes roughly 20-40 minutes on a CPU-only machine (most of it spent on the
llama3.2-vision calls in Task 1 and U-Net training in Task 3).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import task1_data_and_vlm  # noqa: E402
import task2_classical_llm  # noqa: E402
import task3_train_unet  # noqa: E402
import task4_hybrid_pipeline  # noqa: E402
import extension_robustness  # noqa: E402


def main():
    task1_data_and_vlm.main()
    task2_classical_llm.main()
    task3_train_unet.main()
    task4_hybrid_pipeline.main()
    extension_robustness.main()
    print("\nAll tasks complete. See outputs/figures, outputs/json_records, and outputs/*.csv")


if __name__ == "__main__":
    main()
