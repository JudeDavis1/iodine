import warnings
warnings.filterwarnings('ignore')

import torch
import time
import numpy as np

from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from torchvision.transforms import Compose
from torch.utils.data.dataloader import DataLoader

from data import IMGDataset, N_KEYPOINTS
import config


# - Constants
CHANNELS = 1


class Runner:
    
    def __init__(
        self,
        device: str="cpu",
    ):
        self.device = torch.device(device)
        self.model = HandDTTR()
    
    def fit(
        self,
        epochs: int=10,
        lr: float=0.002,
        batch_size: int=32,
        transform: Compose=None,
        max_data: int=5000
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = self.model.to(self.device)
        self.dataset = IMGDataset(transform, max_data=max_data)

        self.model.train()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.lr,
            betas=(0.99, 0.999)
        )
        criterion = LogCoshLoss()

        # use this to make the batch selection easier
        loader = DataLoader(
            self.dataset,
            self.batch_size,
            shuffle=True
        )
        
        try:
            print('[*] Beginning training...')
            for step in (t := tqdm(range(1, self.epochs + 1))):
                optimizer.zero_grad(set_to_none=True)

                # get the next batch from the loader
                img, y = next(iter(loader))
                y = y.flatten()
                output: torch.Tensor = self.model(img.to(self.device)).float()
                loss: torch.Tensor = criterion(output.flatten(), y.to(self.device).float())

                loss.backward()
                optimizer.step()

                t.set_description(f"{loss}")
            self.model.save('HandDTTR.model')
        except KeyboardInterrupt: pass
        print('Saved!')
    
    def predict(self, img: torch.Tensor):
        subject = img.to(self.device).unsqueeze(0)
        output = self.model(subject).cpu().detach().numpy()

        x_norm, y_norm = np.array_split(output.flatten(), 2)
        return (x_norm * config.WIDTH, y_norm * config.HEIGHT)
    
    def evaluate(self, img):
        subject = img.to(self.device).unsqueeze(0)
        reg_img = subject[0].cpu().numpy().transpose(1, 2, 0)
        output = self.model(subject).cpu().detach().numpy()

        output = np.array_split(output.flatten(), 2)
        plt.imshow(reg_img)
        plt.scatter(output[0] * config.WIDTH, output[1] * config.HEIGHT, s=100, c='black')
        plt.show()


class LogCoshLoss(nn.Module):

    def __init__(self): super().__init__()

    def forward(self, prediction, real):
        return torch.mean(torch.log(torch.cosh(prediction - real)))


class HandDTTR(nn.Module):

    def __init__(self):
        super().__init__()

        self._featuremap = 64
        self._kernel_size = 4
        self.feature_extractor = nn.Sequential(
            *self._conv_block(CHANNELS, self._featuremap * 2),
            *self._conv_block(self._featuremap * 2, self._featuremap),
            *self._conv_block(self._featuremap, 4),
        )

        self.regressor = nn.Sequential(
            FlattenExcludeBatchDim(),
            nn.LazyLinear(128),
            nn.Linear(128, 128),
            nn.Linear(128, 2 * N_KEYPOINTS),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.dropout(self.regressor(features))

        return logits

    def save(self, path: str='./model'):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    
    def _conv_block(self, in_dim, out_dim):
        return [
            nn.Conv2d(in_dim, out_dim, self._kernel_size, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        ]


class FlattenExcludeBatchDim(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, 1)
