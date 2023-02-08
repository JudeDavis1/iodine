import sys
import torch

from torchvision import transforms
from torch.utils.data import DataLoader

from data import IMGDataset
from model import Trainer

import config


LR = .0002
EPOCHS = 10
BATCH_SIZE = 128
N_NO_HAND = 12

device = torch.device('mps')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0), (1)),
    transforms.Grayscale(),
    # transforms.Resize((config.WIDTH, config.HEIGHT))
])

def main():
    global model
    print(f'Using {str(device).upper()} backend')

    trainer = Trainer(
        device=str(device),
        transform=transform,
        batch_size=8
    )
    trainer.fit()



if __name__ == '__main__':
    main()


