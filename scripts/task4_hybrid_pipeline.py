
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import hybrid_pipeline  # noqa: E402


def main():
    print("=== Task 4: hybrid pipeline on unseen test images ===")
    df = hybrid_pipeline.run_task4("test")
    print(df.to_string(index=False))
    print("\nSaved per-image JSON records to outputs/json_records/task4_hybrid_records.json")
    print("Saved aggregated CSV to outputs/task4_hybrid_records.csv")


if __name__ == "__main__":
    main()
