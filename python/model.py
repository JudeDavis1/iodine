import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np

from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import Compose
from torch.utils.data.dataloader import DataLoader

from data import IMGDataset, N_KEYPOINTS
import config


# - Constants
CHANNELS = 3


class Runner:
    
    def __init__(
        self,
        device: str="cpu",
    ):
        self.device = torch.device(device)
        self.model = HandDTTR()
        self.params = {
            'epochs': None,
            'lr': None,
            'mse': [],
            'loss': []
        }
    
    def fit(
        self,
        epochs: int=10,
        lr: float=0.002,
        batch_size: int=32,
        transform: Compose=None,
        max_data: int=5000
    ):
        self.params.update({
            'epochs': epochs,
            'lr': lr,
            # we do these later:
            # - 'mse'
            # - 'loss'
        })
        self.model.train()

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = self.model.to(self.device)
        self.dataset = IMGDataset(transform, max_data=max_data)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.lr
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
            for _ in (t := tqdm(range(1, self.epochs + 1))):
                for img, y in loader:
                    optimizer.zero_grad(set_to_none=True)

                    # get the next batch from the loader
                    y = y.float().flatten().to(self.device)
                    output: torch.Tensor = self.model(img.to(self.device)).float().flatten()
                    loss: torch.Tensor = criterion(output, y)

                    loss.backward()
                    optimizer.step()

                    mse = F.mse_loss(output, y)
                    t.set_description(f"Training Loss: {loss:.5f}  MSE: {mse:.5f}")

                    self.params['mse'].append(mse.cpu().detach().numpy())
                    self.params['loss'].append(loss.cpu().detach().numpy())
                self.model.save('HandDTTR.model')
        except KeyboardInterrupt:
            self.model.save('HandDTTR.model')
            print('Saved!')
    
    def predict(self, img: torch.Tensor):
        subject = img.to(self.device).unsqueeze(0)
        output = self.model(subject).cpu().detach().numpy()

        x_norm, y_norm = np.array_split(output.flatten(), 2)
        return (x_norm * config.WIDTH, y_norm * config.HEIGHT)
    
    def plot_train_data(self):
        epoch_range = list(range(self.params['epochs']))
        plt.plot(epoch_range, self.params['loss'])
        plt.plot(epoch_range, self.params['mse'])
        plt.legend(["loss", "MSE"], loc='lower right')

        plt.show()
    
    def evaluate(self, img):
        subject = img.to(self.device).unsqueeze(0)
        reg_img = subject[0].cpu().numpy().transpose(1, 2, 0)
        output = self.model(subject).cpu().detach().numpy()

        output = np.array_split(output.flatten(), 2)
        plt.imshow(reg_img)
        plt.scatter(output[0] * config.WIDTH, output[1] * config.HEIGHT, s=100, c='black')
        plt.show()


class LogCoshLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction, real):
        loss = torch.log(torch.cosh(prediction - real))
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError('`reduction` must be sum or mean')


class HandDTTR(nn.Module):

    def __init__(self):
        super().__init__()

        self._featuremap = 64
        self._kernel_size = 3
        self.out_features = 2 * N_KEYPOINTS
        self.bottleneck_input_size = 128
        self.feature_extractor = nn.Sequential(
            *self._conv_block(CHANNELS, self._featuremap * 4),
            *self._conv_block(self._featuremap * 4, self._featuremap * 2, reduction=False),
            *self._conv_block(self._featuremap * 2, self._featuremap, reduction=False),
            *self._conv_block(self._featuremap, self.out_features, reduction=False),
        )
        self.bottleneck = nn.LazyLinear(self.bottleneck_input_size)
        self.regressor = nn.Sequential(
            nn.Linear(self.bottleneck_input_size, self.bottleneck_input_size * 2),
            nn.Linear(self.bottleneck_input_size * 2, self.out_features),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.flatten(self.feature_extractor(x), 1)
        bottleneck_output = self.bottleneck(features)
        logits = self.regressor(bottleneck_output)

        return logits

    def save(self, path: str='./model'):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
    
    def _conv_block(self, in_dim, out_dim, reduction=True):
        layers = [
            nn.Conv2d(in_dim, out_dim, self._kernel_size, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        ]

        if reduction:
            layers.insert(1, nn.MaxPool2d(2, 2))
        
        return layers

