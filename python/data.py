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
    data = []
    json_data = json.load(open(JSON_ROOT))['annotations']
    # random.shuffle(json_data)

    t = tqdm(range(max_data), leave=True)
    for i, object in enumerate(json_data):
        if i % UPDATE_INTER == 0:
            t.update(UPDATE_INTER)
        
        img_path = os.path.join(IMG_ROOT, str(object['id']) + '.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if transform:
            try:
                img = transform(img)
            except:
                continue

        coords = np.array_split(json_data[i]['keypoints'], N_KEYPOINTS)
        tensor_coords = get_critical_points(coords)

        # plt.imshow(img)
        # plt.scatter(tensor_coords[0].detach().numpy() * config.WIDTH, tensor_coords[1].detach().numpy()* config.HEIGHT)
        # plt.show()

        data_point = [img, tensor_coords]
        data.append(data_point)
        if i >= max_data:
            break

    return data


def get_critical_points(coords) -> torch.Tensor:
    x = []
    y = []

    for pt in coords:
        x.append(float(pt[0]))
        y.append(float(pt[1]))
    
    x = torch.tensor(x) / config.WIDTH
    y = torch.tensor(y) / config.HEIGHT
    return torch.stack([x, y])

def unnormalize(tensor: torch.Tensor, low, high) -> torch.Tensor:
    std = (tensor - low) / (high - low)
    scaled_tensor = std * (high - low) + low
    return scaled_tensor

class IMGDataset(Dataset):

    def __init__(self, transform: torchvision.transforms.Compose=None, max_data=3000):
        self.data = load_data(max_data, transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    load_data()
