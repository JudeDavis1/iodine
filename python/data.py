import os
import sys
import cv2
import torch
import random
import torchvision
import numpy as np

from torch.utils.data import Dataset

import config


def run():
    vc = cv2.VideoCapture(0)

    mode = sys.argv[1]

    i = 0
    while True:
        _, frame = vc.read()
        frame = cv2.cvtColor(cv2.resize(frame, (config.WIDTH, config.HEIGHT)), cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Your frame', frame)

        rnd = random.randbytes(10).hex()
        if mode == 'hand':
            cv2.imwrite(f'data/hand/img{rnd}.jpg', frame)
        elif mode == 'no_hand':
            cv2.imwrite(f'data/no_hand/img{rnd}.jpg', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if i >= 1000:
            break

        i += 1

    vc.release()

def load_data():
    data = []

    label = 1
    for img_file in os.listdir('data/hand'):
        img = cv2.imread(os.path.join('data/hand', img_file), cv2.COLOR_BGR2GRAY)

        data.append((img, label))

    label = 0
    for img_file in os.listdir('data/no_hand'):
        img = cv2.imread(os.path.join('data/no_hand', img_file), cv2.COLOR_BGR2GRAY)

        data.append((img, label))

    return data


class IMGDataset(Dataset):

    def __init__(self, transform: torchvision.transforms.Compose=None):
        if transform:
            self.data = [(transform(img), torch.tensor(label).float()) for img, label in load_data()]
        self.data = load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    run()
    cv2.destroyAllWindows()
