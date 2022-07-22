import albumentations as albu
from albumentations.augmentations import transforms as A
from albumentations.pytorch import ToTensor


def pre_transforms(image_size=512):
    return [albu.Resize(height=1088, width=1920, p=1)]


def hard_transforms():
    result = [
        A.HorizontalFlip(),
        albu.OneOf([
            albu.Cutout(),
            # albu.RandomSizedCrop(min_max_height=(900, 1000), height=1088, width=1920, p=0.1),
            A.RandomGamma(p=0.2),
            albu.ToGray(p=0.2),
            albu.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
            albu.GridDistortion(p=0.3),
            albu.HueSaturationValue(p=0.3)], p=1)
    ]

    return result


def resize_transforms(image_size=512):
    BORDER_CONSTANT = 0
    pre_size = int(image_size * 1.1)

    random_crop = albu.Compose([
        albu.SmallestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])

    random_crop_big = albu.Compose([
        albu.LongestMaxSize(pre_size, p=1),
        albu.RandomCrop(
            image_size, image_size, p=1
        )

    ])

    # Converts the image to a square of size image_size x image_size
    result = [
        albu.OneOf([
            random_crop,
            rescale,
            random_crop_big
        ], p=1)
    ]

    return result


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]


def transpose(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def get_preprocessing():
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=transpose, mask=transpose),
    ]
    return albu.Compose(_transform)
