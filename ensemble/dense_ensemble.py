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
INPUT_PATH = '/home/winky/PycharmProjects/Trian/data/test_images'
OUTPUT_PATH = '/home/winky/PycharmProjects/Trian/data/ensemble/test/segformer'

id2label, label2id, id2color = get_palette()
_, best_feature_extractor = get_model("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")
del _
best_model = torch.load('/home/winky/PycharmProjects/Trian/transformers/Transformers_schedluer_b3/checkpoints/a30_1_0.818236768245697_model.pth')
best_model.to(device)
best_model.eval()


model_list = [[best_model, best_feature_extractor]]
for img_name in tqdm(os.listdir(INPUT_PATH)):
    image = Image.open(os.path.join(INPUT_PATH, img_name))
    segs = None
    img = np.array(image)
    img_mean = np.mean(img)
    model, feature_extractor = model_list[0]
    pixel_values = best_feature_extractor(image, return_tensors="pt").pixel_values.to(device)


    outputs = best_model(pixel_values)
    logits = outputs.logits

    logits = nn.functional.interpolate(outputs.logits.detach().cpu(),
                    size=image.size[::-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)
    seg = torch.sigmoid(logits) * 255
    seg = torch.tensor(seg, dtype=torch.uint8)[0][1:].permute(1, 2, 0)

    # print(seg.numpy().shape)
    cv2.imwrite(os.path.join(OUTPUT_PATH, img_name), seg.numpy())
    # torch.save(logits, os.path.join(OUTPUT_PATH, img_name.replace('.png', '.pt')))
    # seg = logits.argmax(dim=1)[0]
    # color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    #
    # # palette = np.array(ade_palette())
    # for id, color in id2color.items():
    #     color_seg[seg == id, :] = color
    # # Convert to BGR
    # color_seg = color_seg[..., ::-1]
    #
    # # Show image + mask
    # # img = np.array(image) * 0.5 + color_seg * 0.5 * 25
    # img = color_seg
    #
    # img = img.astype(np.uint8)

    # cv2.imwrite(os.path.join(OUTPUT_PATH, img_name), color_seg)
    # img = cv2.resize(img, (1500, 700), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('result', img)
    # cv2.waitKey(0)
