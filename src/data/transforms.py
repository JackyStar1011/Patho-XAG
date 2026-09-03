from PIL import Image
from torchvision import transforms


# ============================================================
# ImageNet normalization
# Used for ImageNet-pretrained models such as ResNet-50
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# ROI augmentation
# ============================================================

def get_train_roi_transform():
    """
    Apply augmentation to the full ROI.

    The same transformation is shared by all patches
    inside one ROI.
    """

    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        # Rotate only by 0, 90, 180, or 270 degrees.
        transforms.RandomChoice([
            transforms.Lambda(
                lambda img: img
            ),
            transforms.Lambda(
                lambda img: img.transpose(
                    Image.Transpose.ROTATE_90
                )
            ),
            transforms.Lambda(
                lambda img: img.transpose(
                    Image.Transpose.ROTATE_180
                )
            ),
            transforms.Lambda(
                lambda img: img.transpose(
                    Image.Transpose.ROTATE_270
                )
            ),
        ]),

        # Mild color changes
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
        ),
    ])


def get_eval_roi_transform():
    """
    No augmentation for validation and test.
    """

    return None


# ============================================================
# Tensor conversion and normalization
# ============================================================

def get_image_transform():
    """
    Convert PIL image to tensor and normalize it.
    """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])