import os
import cv2
import torchvision
import torch
import json
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.nn import functional as F

import config

ROOT = './data'


def load_data(transform):
    critical_key = 'hand_pts'
    files = os.listdir(ROOT)
    max_files = len(files)
    t = tqdm(range(max_files))
    data = []

    for file in files:
        fname = os.path.join('.', ROOT, file)

        if fname.endswith('.json'):
            # the name of the file (without extension)
            name = fname.strip('.json')
            json_data = json.load(open(fname, encoding='utf-8'))
            pts = get_critical_points(json_data, critical_key)

            img = cv2.imread('.' + name + '.jpg')
            data.append([transform(img), pts])
            # plt.imshow(transform(img).detach().numpy().transpose(1, 2, 0))
            # plt.scatter(pts[0] * config.WIDTH, pts[1] * config.HEIGHT, s=100, c='black')
            # plt.show()
        
        t.update(1)
    
    return data


def get_critical_points(json, key) -> Tuple[np.ndarray]:
    x = []
    y = []

    for pt in json[key]:
        x.append(float(pt[0]))
        y.append(float(pt[1]))
    
    x = torch.tensor(x)
    y = torch.tensor(y)
    return torch.stack([x.mean() / config.WIDTH, y.mean() / config.HEIGHT])

class IMGDataset(Dataset):

    def __init__(self, transform: torchvision.transforms.Compose=None):
        self.data = load_data(transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    load_data()
