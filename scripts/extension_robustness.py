
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import robustness  # noqa: E402


def main():
    print("=== Extension: robustness trace on corrupted test images ===")
    results = robustness.run_robustness_trace()
    for r in results:
        print(f"\n{r['corrupted_file']} (base: {r['base_image_id']})")
        print(f"  stage1 pixel mean %% change: {r['stage1_pixel_stats']['mean_pct_change']:.1f}%")
        print(f"  stage2 U-Net Dice drop vs GT: {r['stage2_mask_overlap']['dice_drop']:.3f}")
        print(f"  stage3 n_objects %% change: {r['stage3_feature_table']['n_objects_pct_change']:.1f}%")
    print("\nSaved full trace to outputs/json_records/robustness_trace.json")


if __name__ == "__main__":
    main()
