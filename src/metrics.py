"""Segmentation metrics shared by the U-Net evaluation and the Otsu-vs-U-Net comparison."""
import numpy as np


def dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Dice/F1 overlap between two boolean (or 0/1) masks of the same shape."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return float((2.0 * intersection + eps) / (pred.sum() + gt.sum() + eps))


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-7) -> float:
    """Intersection-over-Union between two boolean (or 0/1) masks of the same shape."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((intersection + eps) / (union + eps))
