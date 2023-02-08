import warnings
warnings.filterwarnings('ignore')

import cv2
import torch

from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from torchvision.transforms import Compose
from torch.utils.data.dataloader import DataLoader

from data import IMGDataset
import config


# - Constants
CHANNELS = 1


class Trainer:
    
    def __init__(
        self,
        epochs: int=10,
        lr: float=0.002,
        device: str="cpu",
        batch_size: int=32,
        transform: Compose=None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)
        self.model = HandDTTR()
        self.dataset = IMGDataset(transform)
    
    def fit(self):
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.lr,
            (0.9, 0.999)
        )
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(
            self.dataset,
            self.batch_size,
            shuffle=True
        )

        t_steps = len(loader)
        try:
            for epoch in (t := tqdm(range(1, self.epochs + 1))):
                for j, (img, y) in enumerate(loader):
                    optimizer.zero_grad()
                    y = y.T.flatten()
                    output: torch.Tensor = self.model(img.to(self.device)).float()
                    loss = criterion(output.flatten(), y.to(self.device).float())


                    loss.backward()
                    optimizer.step()

                    if j % 100 == 0:
                        subject = img[0].to(self.device).unsqueeze(0)
                        reg_img = subject[0].cpu().numpy().transpose(1, 2, 0)
                        pt = self.model(subject)[0].cpu().detach()
                        t.set_description(f"{loss} {j + 1}/{t_steps} {pt}")
                        plt.imshow(reg_img)
                        plt.scatter(pt[0] * config.WIDTH, pt[1] * config.HEIGHT, s=100, c='black')
                        plt.show()
                    t.set_description(f"{loss} {j + 1}/{t_steps}")


                self.model.save('HandDTTR.model')
        except KeyboardInterrupt:
            pass    
        print('Saved!')
    
    def evaluate(self):
        correct = 0
        for (x, y) in self.dataset[:30]:
            output = self.model(x)
            prediction = round(output)
            print(prediction)


class HandDTTR(nn.Module):

    def __init__(self):
        super().__init__()

        self._featuremap = 64
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(CHANNELS, self._featuremap * 2, 4, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self._featuremap * 2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self._featuremap * 2, self._featuremap, 4, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self._featuremap),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self._featuremap, 4, 4, bias=False),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Sequential(
            FlattenExcludeBatchDim(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.dropout = nn.Dropout(.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.regressor(features)

        return logits

    def save(self, path: str='./model'):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))



class FlattenExcludeBatchDim(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, 1)
