"""ImageFolder-based ImageNet dataset for RAR with ADM crop augmentations.

Ported from yrRandAR (train_c2i_imagenet.py). Used for on-the-fly tokenization
(replaces the pretokenized JSONL dataset path).
"""
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .augmentation import center_crop_arr, random_crop_arr


def _build_transform(image_size: int, tokenizer_type: str, is_train: bool, random_crop: bool):
    crop_transform = random_crop_arr if (is_train and random_crop) else center_crop_arr
    ops = [transforms.Lambda(lambda img: crop_transform(img, image_size))]
    if is_train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    # MaskGIT VQ tokenizer expects [0, 1]; LlamaGen VQ expects [-1, 1].
    if tokenizer_type == "llamagen":
        ops.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True))
    return transforms.Compose(ops)


class ImageFolderDataset(ImageFolder):
    """ImageFolder that yields a dict batch compatible with RAR's training loop."""

    def __getitem__(self, index):
        image, class_id = super().__getitem__(index)
        return {"image": image, "class_id": class_id}


def build_image_folder(data_path: str, image_size: int = 256,
                       tokenizer_type: str = "maskgit",
                       is_train: bool = True, random_crop: bool = False) -> ImageFolderDataset:
    transform = _build_transform(image_size, tokenizer_type, is_train, random_crop)
    return ImageFolderDataset(data_path, transform=transform)
