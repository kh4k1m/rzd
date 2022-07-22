from pathlib import Path
from typing import List

import cv2
from catalyst import utils
import numpy as np
from skimage.io import imread as gif_imread
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
            self,
            images: List[Path],
            masks: List[Path] = None,
            transforms=None
    ) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.class_values = [0, 6, 7, 10]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = utils.imread(image_path)

        result = {"image": image}

        if self.masks is not None:
            mask = gif_imread(self.masks[idx])
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = [(mask == v) for v in self.class_values]
            mask = np.stack(mask, axis=-1).astype('float')
            # print('ds, mask shape', mask.shape)
            # mask = np.moveaxis(mask, -1, 0) # np.swapaxes(mask, 2, 0, 1)
            # print('ds, mask shape', mask.shape)
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name
        result["img_size"] = image.shape[:2]
        if self.masks is not None:
            mask = result["mask"]
            mask = mask.squeeze(0)
            mask = mask.permute(2, 0, 1)
            result["mask"] = mask
        # print('ds, mask out shape', mask.shape)

        return result
