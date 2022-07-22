from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from catalyst import utils
import numpy as np
import cv2
from skimage.io import imread as gif_imread
import random


def show_examples(name: str, image: np.ndarray, mask: np.ndarray):
    # plt.figure(figsize=(10, 14))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.title(f"Image: {name}")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(mask)
    # plt.title(f"Mask: {name}")
    full_img = np.hstack([image, mask])
    full_img = cv2.resize(full_img, (2300, 950))
    cv2.imshow(name, full_img)
    cv2.waitKey(0)
    cv2.destroyWindow(name)


def show(index: int, images: List[Path], masks: List[Path], transforms=None) -> None:
    image_path = images[index]
    name = image_path.name

    image = utils.imread(image_path)
    mask = gif_imread(masks[index])

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp["image"]
        mask = temp["mask"]

    show_examples(name, image, mask)


def show_random(images: List[Path], masks: List[Path], transforms=None) -> None:
    length = len(images)
    index = random.randint(0, length - 1)
    show(index, images, masks, transforms)
