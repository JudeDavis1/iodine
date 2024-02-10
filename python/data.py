import json
import os
import random
from functools import lru_cache

import cv2
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

import config

IMG_ROOT = "./data/training/rgb"
JSON_ROOT = "./data/freihand_train.json"
N_KEYPOINTS = 21  # standard for this dataset
UPDATE_INTER = 200


def load_data(max_data: int, transform: Compose = None):
    total_files = len(os.listdir(IMG_ROOT))
    max_data = min(max_data, total_files)
    x = torch.empty((max_data, 1, config.HEIGHT, config.WIDTH), dtype=torch.float32)
    y = torch.empty((max_data, 2, N_KEYPOINTS), dtype=torch.float32)
    json_data = json.load(open(JSON_ROOT))["annotations"]

    t = tqdm(range(max_data), leave=True)
    for i, object in enumerate(json_data[:max_data]):
        if i % UPDATE_INTER == 0:
            t.update(UPDATE_INTER)

        object_id = object["id"]
        keypoints = tuple(object["keypoints"])
        try:
            img, tensor_coords = process_img(object_id, keypoints)
            if transform:
                img = transform(img)
            else:
                img = torch.tensor(img, dtype=torch.float32)
        except Exception as e:
            print(e)
            continue

        x[i] = img
        y[i] = tensor_coords

    return x.float(), y.float()


@lru_cache(maxsize=None)
def process_img_with_cache(object_id: int, keypoints: tuple):
    img_path = os.path.join(IMG_ROOT, str(object_id) + ".jpg")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    coords = np.array_split(np.array(keypoints), N_KEYPOINTS)
    tensor_coords = get_critical_points(coords)

    return img, tensor_coords


def process_img(object_id: int, keypoints: tuple):
    return process_img_with_cache(object_id, keypoints)


def get_critical_points(coords) -> torch.Tensor:
    coords_tensor = torch.tensor(coords, dtype=torch.float32)
    x = coords_tensor[:, 0]
    y = coords_tensor[:, 1]

    return torch.stack([x, y])


class GaussianNormalizer:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def normalize(self, tensor: torch.Tensor):
        return (tensor - self.mean) / self.std

    def denormalize(self, tensor: torch.Tensor):
        return (tensor * self.std) + self.mean


class IMGDataset(Dataset):

    def __init__(self, transform: torchvision.transforms.Compose = None, max_data=3000):
        x, y = load_data(max_data, transform)

        print("Normalizing sequences...")
        y_mean: np.ndarray = y.mean().item()
        y_std: np.ndarray = y.std().item()
        self.y_normalizer = GaussianNormalizer(y_mean, y_std)

        y = self.y_normalizer.normalize(y)

        # save as json
        json.dump(
            {
                "y_mean": y_mean,
                "y_std": y_std,
            },
            open("./data/normalization.json", "w"),
        )

        print("Done!")

        self.data = list(zip(x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    load_data(max_data=1000)
