"""Task 3: train the small U-Net, evaluate with Dice/IoU, and save diagnostic figures."""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import NucleiDataset
from .metrics import dice_score, iou_score
from .unet import BCEDiceLoss, UNet


def get_dataloaders(batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    train_ds = NucleiDataset("train")
    val_ds = NucleiDataset("val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: UNet, loader: DataLoader, device: str) -> dict:
    """Mean Dice and IoU over a loader, plus per-image scores for later inspection."""
    model.eval()
    dices, ious, ids = [], [], []
    for imgs, masks, image_ids in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = torch.sigmoid(model(imgs)) > 0.5
        for p, m, img_id in zip(preds.cpu().numpy(), masks.cpu().numpy(), image_ids):
            dices.append(dice_score(p, m))
            ious.append(iou_score(p, m))
            ids.append(img_id)
    return {
        "mean_dice": float(np.mean(dices)),
        "mean_iou": float(np.mean(ious)),
        "per_image": [{"image_id": i, "dice": d, "iou": u} for i, d, u in zip(ids, dices, ious)],
    }


def train(epochs: int = 40, lr: float = 1e-3, batch_size: int = 4, seed: int = config.RANDOM_SEED) -> dict:
    """Train the U-Net, tracking train loss and val Dice per epoch. Saves the best checkpoint
    (by val Dice) and the loss/Dice curves, and returns a results dict."""
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = get_dataloaders(batch_size)
    model = UNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_dice": [], "val_iou": []}
    best_dice = -1.0
    best_path = config.CHECKPOINT_DIR / "unet_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_dice"].append(val_metrics["mean_dice"])
        history["val_iou"].append(val_metrics["mean_iou"])

        if val_metrics["mean_dice"] > best_dice:
            best_dice = val_metrics["mean_dice"]
            torch.save(model.state_dict(), best_path)

        print(f"epoch {epoch:3d}/{epochs}  train_loss={history['train_loss'][-1]:.4f}  "
              f"val_dice={history['val_dice'][-1]:.4f}  val_iou={history['val_iou'][-1]:.4f}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_val_metrics = evaluate(model, val_loader, device)

    plot_curves(history)
    plot_prediction_panels(model, val_loader, device)

    results = {
        "device": device,
        "epochs": epochs,
        "best_val_dice": best_dice,
        "final_val_metrics": final_val_metrics,
        "history": history,
        "checkpoint_path": str(best_path),
    }
    (config.JSON_DIR / "task3_unet_results.json").write_text(json.dumps(results, indent=2))
    return results


def plot_curves(history: dict, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], color="steelblue")
    axes[0].set_title("Training loss (BCE + Dice)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(history["val_dice"], label="Val Dice", color="seagreen")
    axes[1].plot(history["val_iou"], label="Val IoU", color="darkorange")
    axes[1].set_title("Validation Dice / IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    save_path = save_path or config.FIGURES_DIR / "task3_loss_dice_curves.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


@torch.no_grad()
def plot_prediction_panels(model: UNet, loader: DataLoader, device: str, n: int = 3, save_path=None):
    """Save input / ground-truth / prediction panels for `n` validation images."""
    model.eval()
    imgs, masks, ids = next(iter(loader))
    imgs, masks = imgs.to(device), masks.to(device)
    preds = torch.sigmoid(model(imgs)) > 0.5

    n = min(n, imgs.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i in range(n):
        axes[i, 0].imshow(imgs[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 0].set_title(f"{ids[i]} - input")
        axes[i, 1].imshow(masks[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 1].set_title("ground truth")
        axes[i, 2].imshow(preds[i, 0].cpu().numpy(), cmap="gray")
        axes[i, 2].set_title("prediction")
        for j in range(3):
            axes[i, j].axis("off")

    fig.tight_layout()
    save_path = save_path or config.FIGURES_DIR / "task3_prediction_panels.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    results = train(epochs=40)
    print(f"best_val_dice={results['best_val_dice']:.4f}  "
          f"final_val_iou={results['final_val_metrics']['mean_iou']:.4f}")
