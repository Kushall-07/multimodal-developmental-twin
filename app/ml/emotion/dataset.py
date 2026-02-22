import torch
from torchvision import datasets, transforms
from .config import TRAIN_DIR, TEST_DIR, IMG_SIZE


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


def build_datasets():
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=get_transforms(train=True))
    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=get_transforms(train=False))
    return train_ds, test_ds
