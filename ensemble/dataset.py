from torch.utils.data import DataLoader
import os
from PIL import Image
from pathlib import Path
from typing import List
import random

import cv2
# from catalyst import utils
import numpy as np
from skimage.io import imread as gif_imread
from torch.utils.data import Dataset

from utils import get_palette



class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train
        self.class_values = [0, 6, 7, 10]

        sub_path = "train" if self.train else "val"
        self.img_dir = os.path.join(self.root_dir, "images")  # , sub_path)
        self.ann_dir = os.path.join(self.root_dir, "mask")  # , sub_path)

        # read images
        image_file_names = []
        val_image_file_names = []
        for file_name in os.listdir(self.img_dir):
            if '0.4' not in file_name:
                image_file_names.append(file_name)
            else:
                val_image_file_names.append(file_name)
        # for root, dirs, files in os.walk(self.img_dir):
        #     image_file_names.extend(files)

        # read annotations
        annotation_file_names = []
        val_annotation_file_names = []
        for file_name in os.listdir(self.ann_dir):
            if '0.4' not in file_name:
                annotation_file_names.append(file_name)
            else:
                val_annotation_file_names.append(file_name)
        # for root, dirs, files in os.walk(self.ann_dir):
        #     annotation_file_names.extend(files)
        if train:
            self.images = sorted(image_file_names)
            self.annotations = sorted(annotation_file_names)
        else:
            self.images = sorted(val_image_file_names)
            self.annotations = sorted(val_annotation_file_names)
        print(len(self.annotations), len(self.images))
        id2label, label2id, id2color = get_palette()
        self.id2color = id2color
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        annotation = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))


        # make 2D segmentation map (based on 3D one)
        # thanks a lot, Stackoverflow: https://stackoverflow.com/questions/61897492/finding-the-number-of-pixels-in-a-numpy-array-equal-to-a-given-color
        annotation = np.array(annotation)
        annotation_2d = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.uint8)  # height, width

        for id, color in self.id2color.items():
            annotation_2d[(annotation == color).all(axis=-1)] = id

        # randomly crop + pad both image and segmentation map to same size
        # feature extractor will also reduce labels!
        encoded_inputs = self.feature_extractor(
            Image.fromarray(image) if self.transform is not None else image,
            Image.fromarray(annotation_2d),
            return_tensors="pt")
        # print(encoded_inputs["pixel_values"].shape)
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


def get_dataloader(root_dir, feature_extractor, train_batch_size=8, val_batch_size=8):
    train_dataset = SemanticSegmentationDataset(root_dir, feature_extractor, train=True)
    val_dataset = SemanticSegmentationDataset(root_dir, feature_extractor, train=False)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, num_workers=12)
    eval_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=12)
    return train_dataloader, eval_dataloader


if __name__ == '__main__':
    from model import get_model

    root_dir = '/home/winky/PycharmProjects/Trian/data'

    model, feature_extractor = get_model()
    train_dataset = SemanticSegmentationDataset(root_dir, feature_extractor, train=True)
    for elem in train_dataset:
        print(elem.keys())
        break
    print(len(train_dataset))
