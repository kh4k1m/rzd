from typing import List
from pathlib import Path
import torch
import catalyst
import numpy as np
from catalyst import utils
import collections
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
from catalyst.contrib.nn.criterion import DiceLoss, IoULoss
from torch import optim
from catalyst.contrib.nn import RAdam, Lookahead
from catalyst.dl import DiceCallback, IouCallback, \
    CriterionCallback, MetricAggregationCallback, OptimizerCallback
from catalyst.contrib.callbacks import DrawMasksCallback
from catalyst.contrib.nn.schedulers import OneCycleLRWithWarmup


from data.dataset import SegmentationDataset
from data.augmentation import resize_transforms, hard_transforms, post_transforms, pre_transforms, compose
from visualize import show_examples, show_random,show
from model.model import get_model

print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")

SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

# ROOT = Path("/home/shkeep/datasets/salim/512x512/")
ROOT = Path("/home/winky/PycharmProjects/Trian/data/")


train_image_path = ROOT / "images"
train_mask_path = ROOT / "mask"


ALL_IMAGES = sorted(train_image_path.glob("*.png"))
len(ALL_IMAGES)

ALL_MASKS = sorted(train_mask_path.glob("*.png"))
len(ALL_MASKS)

# show_random(ALL_IMAGES, ALL_MASKS)





train_transforms = compose([
    pre_transforms(),
    #resize_transforms(),
    hard_transforms(),
    post_transforms()
])
valid_transforms = compose([pre_transforms(), post_transforms()])

show_transforms = compose([resize_transforms(), hard_transforms()])

# show_random(ALL_IMAGES, ALL_MASKS, transforms=show_transforms)


def get_loaders(
        images: List[Path],
        masks: List[Path],
        random_state: int,
        valid_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transforms_fn=None,
        valid_transforms_fn=None,
) -> dict:
    indices = np.arange(len(images))

    # Let's divide the data set into train and valid parts.
    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_size, random_state=random_state, shuffle=True
    )

    np_images = np.array(images)
    np_masks = np.array(masks)

    # Creates our train dataset
    train_dataset = SegmentationDataset(
        images=np_images[train_indices].tolist(),
        masks=np_masks[train_indices].tolist(),
        transforms=train_transforms_fn
    )

    # Creates our valid dataset
    valid_dataset = SegmentationDataset(
        images=np_images[valid_indices].tolist(),
        masks=np_masks[valid_indices].tolist(),
        transforms=valid_transforms_fn
    )

    # Catalyst uses normal torch.data.DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    # And excpect to get an OrderedDict of loaders
    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


batch_size = 2

print(f"batch_size: {batch_size}")

loaders = get_loaders(
    images=ALL_IMAGES,
    masks=ALL_MASKS,
    random_state=SEED,
    train_transforms_fn=train_transforms,
    valid_transforms_fn=valid_transforms,
    batch_size=batch_size
)
# for x in loaders['valid']:
#     print(x['image'].shape, x['mask'].shape)
# exit(0)


model = get_model()


# we have multiple criterions
criterion = {
    "dice": DiceLoss(),
    "iou": IoULoss(),
    "bce": nn.BCEWithLogitsLoss()
}



learning_rate = 0.001
encoder_learning_rate = 0.0005

# Since we use a pre-trained encoder, we will reduce the learning rate on it.
layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}

# This function removes weight_decay for biases and applies our layerwise_params
model_params = utils.process_model_params(model, layerwise_params=layerwise_params)

# Catalyst has new SOTA optimizers out of box
base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
optimizer = Lookahead(base_optimizer)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)
scheduler = OneCycleLRWithWarmup(
    optimizer,
    num_steps=5,
    lr_range=(0.001, 0.000001),
    warmup_steps=5,
    momentum_range=(0.85, 0.95))
from catalyst.dl import SupervisedRunner

num_epochs = 100
logdir = "./logs/segmentation"

device = utils.get_device()
print(f"device: {device}")

# fp16_params = dict(opt_level="O1") # params for FP16
fp16_params = None

print(f"FP16 params: {fp16_params}")

# by default SupervisedRunner uses "features" and "targets",
# in our case we get "image" and "mask" keys in dataset __getitem__
runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")


callbacks = [
    # Each criterion is calculated separately.
    CriterionCallback(
        input_key="mask",
        prefix="loss_dice",
        criterion_key="dice"
    ),
    CriterionCallback(
        input_key="mask",
        prefix="loss_iou",
        criterion_key="iou"
    ),
    CriterionCallback(
        input_key="mask",
        prefix="loss_bce",
        criterion_key="bce"
    ),

    # And only then we aggregate everything into one loss.
    MetricAggregationCallback(
        prefix="loss",
        mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
        # because we want weighted sum, we need to add scale for each loss
        metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
    ),

    # metrics
    DiceCallback(input_key="mask"),
    IouCallback(input_key="mask"),
    OptimizerCallback(accumulation_steps=16),
    # visualization
    DrawMasksCallback(output_key='logits',
                      input_image_key='image',
                      input_mask_key='mask',
                      summary_step=50
                      )
]

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    # our dataloaders
    loaders=loaders,
    # We can specify the callbacks list for the experiment;
    callbacks=callbacks,
    # path to save logs
    logdir=logdir,
    num_epochs=num_epochs,
    # save our best checkpoint by IoU metric
    main_metric="iou",
    # IoU needs to be maximized.
    minimize_metric=False,
    # for FP16. It uses the variable from the very first cell
    fp16=True,
    # prints train logs
    verbose=True,
)


