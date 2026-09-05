import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)


def create_resnet50(
    num_classes=2,
    pretrained=True,
):
    """
    Create a ResNet-50 model for CAMEL classification.

    Classes:
        0 = Normal
        1 = Adenoma
    """

    # Load ImageNet pretrained weights
    if pretrained:
        weights = ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = resnet50(weights=weights)

    # Get the input size of the original classifier
    num_features = model.fc.in_features

    # Replace the ImageNet classifier
    # Original: 1000 classes
    # New: 2 classes
    model.fc = nn.Linear(
        num_features,
        num_classes,
    )

    return model


if __name__ == "__main__":

    model = create_resnet50()

    print(model)

    # Small test input
    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(x)

    print("\nInput shape :", x.shape)
    print("Output shape:", output.shape)