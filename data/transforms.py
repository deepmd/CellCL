import random

import numpy as np
import torchvision.transforms as torch_transforms
import albumentations as transforms
from albumentations.pytorch import ToTensorV2


class TwoViewsTransform:
    """Take two views of one image as the query and key."""

    def __init__(self, q_transform, k_transform=None):
        self.q_transform = q_transform
        self.k_transform = k_transform or q_transform

    def __call__(self, x):
        q = self.q_transform(image=np.array(x))["image"]
        k = self.k_transform(image=np.array(x))["image"]
        return [q, k]


class RandomCenterCrop(object):

    def __init__(self, scale=(0.08, 1.0)):
        self.scale = scale

    def __call__(self, sample):
        w, h = sample.size
        new_half_size = int(np.ceil(random.uniform(self.scale[0], self.scale[1]) * min(h, w) / 2))

        c_h = h // 2
        c_w = w // 2

        cropped = sample.crop((c_w - new_half_size, c_h - new_half_size,
                               c_w + new_half_size, c_h + new_half_size))

        return cropped


def get_train_transform(augmentation):
    image_size = 96

    aug_strong = transforms.Compose([
        transforms.RandomResizedCrop(image_size, image_size),
        # transforms.Compose([RandomCenterCrop(), transforms.Resize(height=image_size, width=image_size)]),
        transforms.HorizontalFlip(),
        transforms.ElasticTransform(),
        # transforms.RandomRotate90(),
        transforms.ColorJitter(p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.Compose([RandomCenterCrop(), transforms.Resize(height=image_size, width=image_size)]),
        # transforms.Compose([transforms.CenterCrop(image_size//2), transforms.Resize(height=image_size, width=image_size)]),
        transforms.GaussianBlur(blur_limit=[int(0.1 * image_size), int(0.1 * image_size)], sigma_limit=[0.1, 2]),
        transforms.Normalize(mean=(0.7469, 0.7403, 0.7307), std=(0.1548, 0.1594, 0.1706)),
        ToTensorV2()
    ])

    aug_weak = transforms.Compose([
        transforms.RandomResizedCrop(image_size, image_size),
        transforms.HorizontalFlip(),
        transforms.Normalize(mean=(0.7469, 0.7403, 0.7307), std=(0.1548, 0.1594, 0.1706)),
        ToTensorV2()
    ])

    if augmentation == "weak/strong":
        return TwoViewsTransform(k_transform=aug_weak, q_transform=aug_strong)
    elif augmentation == "weak/weak":
        return TwoViewsTransform(k_transform=aug_weak, q_transform=aug_weak)
    elif augmentation == "strong/weak":
        return TwoViewsTransform(k_transform=aug_strong, q_transform=aug_weak)
    else:  # strong/strong
        return TwoViewsTransform(k_transform=aug_strong, q_transform=aug_strong)


def get_eval_transform():
    composed = torch_transforms.Compose([
        torch_transforms.ToTensor(),
        torch_transforms.Normalize(mean=[0.7469, 0.7403, 0.7307], std=[0.1548, 0.1594, 0.1706])
    ])
    return composed
