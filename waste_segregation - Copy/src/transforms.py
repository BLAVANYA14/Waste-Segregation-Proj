"""Transforms module providing torchvision and albumentations augmentations.
Usage: from src.transforms import get_transforms
       train_tf = get_transforms(backend='torchvision', split='train')
       val_tf = get_transforms(backend='albumentations', split='val')
"""
from typing import Callable

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)


def get_torchvision_transforms(split: str = 'train', img_size: int = 224):
    import torchvision.transforms as T

    if split == 'train':
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.RandomPerspective(distortion_scale=0.2, p=0.3),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    else:
        return T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def get_albumentations_transforms(split: str = 'train', img_size: int = 224):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if split == 'train':
        return A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.5),
            A.ToGray(p=0.1),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(p=0.3),
            ], p=0.3),
            A.Perspective(scale=(0.02, 0.1), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=img_size//16, max_width=img_size//16, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


def get_transforms(backend: str = 'torchvision', split: str = 'train', img_size: int = 224) -> Callable:
    """Return a transform callable. backend: 'torchvision' or 'albumentations'."""
    backend = backend.lower()
    if backend == 'torchvision':
        return get_torchvision_transforms(split=split, img_size=img_size)
    elif backend in ('albumentations', 'a'):
        return get_albumentations_transforms(split=split, img_size=img_size)
    else:
        raise ValueError('Unknown backend: ' + backend)
