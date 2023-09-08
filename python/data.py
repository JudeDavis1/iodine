import os
import cv2
import torchvision
import torch
import json
import random
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from tqdm import tqdm

import config


IMG_ROOT = './data/training/rgb'
JSON_ROOT = './data/freihand_train.json'
N_KEYPOINTS = 21  # standard for this dataset
UPDATE_INTER = 200


def load_data(max_data, transform=None):
    total_files = len(os.listdir(IMG_ROOT))
    max_data = min(max_data, total_files)
    x = torch.empty((
        max_data,
        3,
        config.HEIGHT,
        config.WIDTH,
    ), dtype=torch.float32)
    y = torch.empty((
        max_data,
        2,
        N_KEYPOINTS,
    ), dtype=torch.float32)
    json_data = json.load(open(JSON_ROOT))['annotations']

    t = tqdm(range(max_data), leave=True)
    for i, object in enumerate(json_data[:max_data]):
        if i % UPDATE_INTER == 0:
            t.update(UPDATE_INTER)
        
        img_path = os.path.join(IMG_ROOT, str(object['id']) + '.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if transform:
            try:
                img = transform(img)
            except Exception as e:
                print(e)
                continue

        coords = np.array_split(json_data[i]['keypoints'], N_KEYPOINTS)
        tensor_coords = get_critical_points(coords)

        # plt.imshow(img.permute(1, 2, 0))
        # plt.scatter(tensor_coords[0].detach().numpy() * config.WIDTH, tensor_coords[1].detach().numpy() * config.HEIGHT)
        # plt.show()

        # x.append(torch.tensor(img))
        # y.append(tensor_coords)
        x[i] = torch.tensor(img)
        y[i] = tensor_coords

    return x.float(), y.float()


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

    def __init__(self, transform: torchvision.transforms.Compose=None, max_data=3000):
        x, y = load_data(max_data, transform)

        print("Normalizing sequences...")
        y_mean: np.ndarray = y.mean().item()
        y_std: np.ndarray = y.std().item()
        self.y_normalizer = GaussianNormalizer(y_mean, y_std)

        y = self.y_normalizer.normalize(y)

        # save as json
        json.dump({
            'y_mean': y_mean,
            'y_std': y_std,
        }, open('./data/normalization.json', 'w'))

        print("Done!")

        self.data = list(zip(x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    load_data()
