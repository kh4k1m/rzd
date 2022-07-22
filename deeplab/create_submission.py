import os.path
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from catalyst.dl import SupervisedRunner
from catalyst import utils
import cv2
from tqdm import tqdm

from monai import config
from monai.data import create_test_image_2d, list_data_collate
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AddChannel, AsDiscrete, Compose, ScaleIntensity, EnsureType


from data.dataset import SegmentationDataset
from data.augmentation import post_transforms, pre_transforms, compose
from model.model import get_model
from visualize import show_examples

ROOT = Path("/home/winky/PycharmProjects/Trian/data/")
test_image_path = ROOT / "test_images"
TEST_IMAGES = sorted(test_image_path.glob("*.png"))
logdir = "./logs/segmentation"

valid_transforms = compose([post_transforms()])  # pre_transforms(),
# create test dataset
test_dataset = SegmentationDataset(
    TEST_IMAGES,
    transforms=valid_transforms
)

num_workers: int = 12

infer_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=num_workers
)
runner = SupervisedRunner(device='cuda', input_key="image", input_target_key="mask")
model = get_model()
# print(torch.load('logs/segmentation/checkpoints/train.3.pth').keys())
model.load_state_dict(torch.load('logs/segmentation/checkpoints/best.pth')['model_state_dict'])
post_trans = Compose([EnsureType(), Activations(sigmoid=True)]) #, AsDiscrete(threshold=0.5)])
# this get predictions for the whole loader
# predictions = np.vstack(list(map(
#     lambda x: x["logits"].cpu().numpy(),
#     runner.predict_loader(model=model, loader=infer_loader)
# )))

# print(type(predictions))
# print(predictions.shape)
model = model.cuda()
model.eval()
threshold = 0.5
max_count = 5
scaler = torch.cuda.amp.GradScaler()
stck = set()
for i, features in tqdm(enumerate(test_dataset)):
    with torch.no_grad():
        # image = utils.tensor_to_ndimage(features["image"])
        img_size = features['img_size']
        img = features['image'].cuda()
        print(img.shape)
        img = img.unsqueeze(0)
        # cv2.imshow('image', image)
        roi_size = (1920, 1088)
        sw_batch_size = 1
        with torch.cuda.amp.autocast():
            mask = sliding_window_inference(img, roi_size, sw_batch_size, model, overlap=0.75)
        mask = [post_trans(i).detach().cpu().numpy() for i in decollate_batch(mask)]
        print(len(mask), mask[0].shape)
        # mask_list =[]
        # for k in range(4):
        #     mask = torch.from_numpy(logits[k]).sigmoid()
        #     mask = utils.detach(mask > threshold).astype("float")
        #     mask_list.append(mask)

        # mask = np.expand_dims(mask, axis=-1)
        # print(mask)
        # print()
        mask = mask[0]
        class_values = [0, 6, 7, 10]
        mask = np.argmax(mask, axis=0)
        mask[mask == 1] = 6
        mask[mask == 2] = 7
        mask[mask == 3] = 10

        # for j in range(len(class_values)):
        #     mask[j] = mask[j] * class_values[j]
        mask = np.array(mask, dtype=np.uint8)

        # mask = np.max(mask, axis=0)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # print(mask.shape)
        mask = cv2.resize(mask, img_size[::-1], interpolation=cv2.INTER_NEAREST)
        print(np.unique(mask), mask.shape)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)


        filename = features['filename']
        output_path = os.path.join('output', filename)
        cv2.imwrite(output_path, mask)
print(stck)
