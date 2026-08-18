
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import config


def list_image_ids(split: str) -> list[str]:
    """Return sorted image ids (filename stems) for a split, e.g. 'train_000'."""
    img_dir = config.DATA_DIR / split / "images"
    return sorted(p.stem for p in img_dir.glob("*.png"))


def load_and_preprocess(split: str, image_id: str, size: int = config.IMG_SIZE) -> np.ndarray:
    """Load one raw RGB PNG, convert to grayscale, resize to `size`x`size`.

    Returns a uint8 array of shape (size, size).
    """
    path = config.DATA_DIR / split / "images" / f"{image_id}.png"
    img = Image.open(path).convert("L")  # grayscale
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def preprocess_split(split: str, size: int = config.IMG_SIZE) -> dict[str, np.ndarray]:
    """Preprocess every image in a split. Returns {image_id: grayscale array}."""
    return {img_id: load_and_preprocess(split, img_id, size) for img_id in list_image_ids(split)}


def plot_sample_grid(images: dict[str, np.ndarray], n: int = 9, save_path: Path | None = None,
                      seed: int = config.RANDOM_SEED) -> Path:
    """Save a grid of `n` randomly sampled preprocessed images with their ids as titles."""
    rng = np.random.default_rng(seed)
    ids = sorted(images.keys())
    chosen = rng.choice(ids, size=min(n, len(ids)), replace=False)

    n_cols = 3
    n_rows = int(np.ceil(len(chosen) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax, img_id in zip(axes, chosen):
        ax.imshow(images[img_id], cmap="gray", vmin=0, vmax=255)
        ax.set_title(img_id, fontsize=9)
        ax.axis("off")
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Sample preprocessed images (grayscale, 256x256)")
    fig.tight_layout()

    save_path = save_path or config.FIGURES_DIR / "task1_sample_grid.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_intensity_histogram(images: dict[str, np.ndarray], save_path: Path | None = None) -> Path:
    """Save a histogram of pixel intensities pooled across all images in the split."""
    all_pixels = np.concatenate([img.ravel() for img in images.values()])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(all_pixels, bins=64, range=(0, 255), color="steelblue", edgecolor="none")
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"Intensity histogram, pooled over {len(images)} images")
    fig.tight_layout()

    save_path = save_path or config.FIGURES_DIR / "task1_intensity_histogram.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def run_eda(split: str = "train") -> dict:
    """Preprocess a split and save the sample grid + intensity histogram figures.

    Returns a small dict of summary stats, useful for printing/logging.
    """
    images = preprocess_split(split)
    grid_path = plot_sample_grid(images)
    hist_path = plot_intensity_histogram(images)

    all_pixels = np.concatenate([img.ravel() for img in images.values()])
    summary = {
        "split": split,
        "n_images": len(images),
        "image_shape": next(iter(images.values())).shape,
        "mean_intensity": float(all_pixels.mean()),
        "std_intensity": float(all_pixels.std()),
        "sample_grid_path": str(grid_path),
        "histogram_path": str(hist_path),
    }
    return summary


if __name__ == "__main__":
    summary = run_eda("train")
    print(summary)
