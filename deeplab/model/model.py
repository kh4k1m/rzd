import segmentation_models_pytorch as smp

print(smp.__version__)
def get_model():
    # # We will use Feature Pyramid Network with pre-trained ResNeXt50 backbone
    # model = smp.FPN(encoder_name="resnext50_32x4d", classes=4)
    # model = smp.Unet(encoder_name='resnet34', classes=4)


    model = smp.DeepLabV3Plus(encoder_name='resnext50_32x4d', classes=4)

    return model


if __name__ == '__main__':
    import torch

    model = get_model()
    mask = model(torch.ones([8, 3, 224, 224]))
    print(mask)
