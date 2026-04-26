"""Image classification helpers using pretrained torchvision models."""

from functools import lru_cache
from typing import Dict
from typing import List

import torch
from PIL import Image
from torchvision import models


@lru_cache(maxsize=1)
def _load_classifier_components():
    """Load pretrained classifier, preprocessing transform, and labels once."""
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.eval()

        preprocess = weights.transforms()
        labels = weights.meta["categories"]
        return model, preprocess, labels
    except Exception as exc:
        raise RuntimeError(
            "Failed to load pretrained ResNet18 model. "
            "Check internet access for first-time weight download."
        ) from exc


def classify_image(image: Image.Image, top_k: int = 3) -> List[Dict[str, float]]:
    """Return top-k ImageNet predictions with confidence scores."""
    model, preprocess, labels = _load_classifier_components()

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits[0], dim=0)

    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    results: List[Dict[str, float]] = []
    for probability, index in zip(top_probs, top_indices):
        results.append(
            {
                "class_name": labels[index.item()],
                "confidence": float(probability.item()),
            }
        )

    return results
