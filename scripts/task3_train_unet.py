
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import otsu_vs_unet, train_unet  # noqa: E402


def main():
    print("=== Task 3a: train U-Net ===")
    results = train_unet.train(epochs=40)
    print(f"\nBest val Dice: {results['best_val_dice']:.4f}")
    print(f"Final val IoU: {results['final_val_metrics']['mean_iou']:.4f}")

    print("\n=== Task 3b: Otsu vs U-Net comparison (val split) ===")
    comparison = otsu_vs_unet.compare_on_split("val")
    print(f"Mean Otsu Dice: {comparison['mean_otsu_dice']:.4f}")
    print(f"Mean U-Net Dice: {comparison['mean_unet_dice']:.4f}")
    print(f"Example where U-Net wins most: {comparison['example_unet_wins']['image_id']}")
    print(f"Example where Otsu wins most: {comparison['example_otsu_wins']['image_id']}")


if __name__ == "__main__":
    main()
