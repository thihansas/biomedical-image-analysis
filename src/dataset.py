
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config, data_prep


class NucleiDataset(Dataset):
    """Loads grayscale images and their 0/255 binary masks for one split.

    Images are normalised to [0, 1]; masks are binarised to {0, 1} floats.
    Both are returned as (1, H, W) float32 tensors.
    """

    def __init__(self, split: str, image_ids: list[str] | None = None):
        self.split = split
        self.image_ids = image_ids or data_prep.list_image_ids(split)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img = data_prep.load_and_preprocess(self.split, image_id)  # uint8 (H, W)
        mask_path = config.DATA_DIR / self.split / "masks" / f"{image_id}.png"
        mask = np.array(Image.open(mask_path).convert("L"))

        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return img_tensor, mask_tensor, image_id
