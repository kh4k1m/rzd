import os

from PIL import Image
import torch
from torch import nn
import numpy as np
import cv2
from tqdm import tqdm

from model import get_model
from utils import get_palette

device = 'cuda'
# INPUT_PATH = '/home/winky/PycharmProjects/Trian/data/test_images/'
INPUT_PATH = '/home/winky/PycharmProjects/Trian/data/Dima_tesy/input/'
# OUTPUT_PATH = 'output_transferlearning'
OUTPUT_PATH = '/home/winky/PycharmProjects/Trian/data/ensemble/test/output_1_epoch'

id2label, label2id, id2color = get_palette()
_, feature_extractor = get_model()
del _
model = torch.load('checkpoints/a30_3_0.8182366490364075_model.pth')
model.to(device)
model.eval()
for img_name in tqdm(os.listdir(INPUT_PATH)):
    image = Image.open(os.path.join(INPUT_PATH, img_name))
    if np.array(image).shape[-1] != 3:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
        image = Image.fromarray(image)
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)


    outputs = model(pixel_values)
    logits = outputs.logits

    logits = nn.functional.interpolate(outputs.logits.detach().cpu(),
                    size=image.size[::-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)

    seg = logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

    # palette = np.array(ade_palette())
    for id, color in id2color.items():
        color_seg[seg == id, :] = color
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    # img = np.array(image) * 0.5 + color_seg * 0.5 * 25
    img = color_seg

    img = img.astype(np.uint8)

    cv2.imwrite(os.path.join(OUTPUT_PATH, img_name), color_seg)
    # img = cv2.resize(img, (1500, 700), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('result', img)
    # cv2.waitKey(0)
