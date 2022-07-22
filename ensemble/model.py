import numpy as np
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
from utils import get_palette



# def get_model(model_name="nvidia/mit-b4,"):
def get_model(model_name="nvidia/segformer-b3-finetuned-cityscapes-1024-1024"):
    id2label, label2id, id2color = get_palette()
    # confid = SegformerConfig()
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name,
                                                             num_labels=4,
                                                             id2label=id2label,
                                                             label2id=label2id,
                                                             ignore_mismatched_sizes=True)
    return model, feature_extractor


if __name__ == '__main__':
    img = np.random.randint(low=0, high=255, size=[3, 1024, 1024], dtype=np.uint8)
    print(img.shape)
    model, feature_extractor = get_model()

    inputs = feature_extractor(images=img, return_tensors="pt")
    mask = model(**inputs)
    print(mask.logits.shape)
