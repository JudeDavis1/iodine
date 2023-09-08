import warnings
warnings.filterwarnings('ignore')

import os
import torch
import numpy as np

from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.transforms import Compose
from torch.utils.data.dataloader import DataLoader

from data import IMGDataset, N_KEYPOINTS, GaussianNormalizer

import config


# - Constants
CHANNELS = 3


class Runner:
    
    def __init__(
        self,
        device: str="cpu",
        dropout: float=0.1,
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
        self.dataset = IMGDataset(transform, max_data=max_data)
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

        train_data_amount = int(len(self.dataset) * 0.99)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.lr
        )

        if os.path.exists('optimizer.pt'):
            optimizer.load_state_dict(torch.load('optimizer.pt'))
        criterion = nn.MSELoss()

        # use this to make the batch selection easier
        loader = DataLoader(
            self.dataset[:train_data_amount],
            self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            self.dataset[train_data_amount:],
            self.batch_size,
            shuffle=True
        )
        n_steps = len(loader)
        
        try:
            print('[*] Beginning training...')
            for _ in (t := tqdm(range(1, self.epochs + 1))):
                for j, (img, y) in enumerate(loader):
                    optimizer.zero_grad(set_to_none=True)

                    # get the next batch from the loader
                    y = y.float().flatten().to(self.device)
                    output: torch.Tensor = self.model(img.to(self.device)).float().flatten()
                    loss: torch.Tensor = criterion(output, y)

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    if j % 5 == 0:
                        self.model.eval()
                        img_val, y_val = next(iter(val_loader))
                        y_val = y_val.float().flatten().to(self.device)
                        output_val: torch.Tensor = self.model(img_val.to(self.device)).float().flatten()
                        loss_val: torch.Tensor = criterion(output_val, y_val)
                        self.model.train()
                    
                    t.set_description(f"Training Loss: {loss:.5f}  Validation Loss: {loss_val:.5f} {j + 1}/{n_steps}")

                self.params['loss'].append(loss.item())

                self.model.save('HandDTTR.model')
                torch.save(optimizer.state_dict(), 'optimizer.pt')
        except KeyboardInterrupt:
            self.model.save('HandDTTR.model')
            print('Saved!')
    
    def predict(self, img: torch.Tensor, y_normalizer: GaussianNormalizer):
        subject = img.to(self.device).unsqueeze(0)
        output = self.model(subject).cpu().detach().numpy()

        x_norm, y_norm = np.array_split(output.flatten(), 2)
        return (
            y_normalizer.denormalize(x_norm),
            y_normalizer.denormalize(y_norm),
        )
    
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


# class HandDTTR(nn.Module):

#     def __init__(self, dropout=0.1):
#         super().__init__()

#         self._featuremap = 32
#         self._kernel_size = 3
#         self.dropout = dropout
#         self.out_features = 2 * N_KEYPOINTS
#         self.bottleneck_input_size = 64
#         self.feature_extractor = nn.Sequential(
#             ResidualBlock(CHANNELS, self._featuremap, stride=1),
#             ResidualBlock(self._featuremap, self._featuremap * 2, stride=1),
#             ResidualBlock(self._featuremap * 2, self._featuremap * 4, stride=2),
#             ResidualBlock(self._featuremap * 4, self._featuremap * 8, stride=2),
#         )
#         self.regressor = nn.Sequential(
#             nn.Linear(self._get_conv_output_shape(self.feature_extractor), self.bottleneck_input_size),
#             nn.Dropout(dropout),
#         )

#         self.head = nn.Linear(self.bottleneck_input_size, self.out_features)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         features: torch.Tensor = self.feature_extractor(x)
#         features = features.view(features.shape[0], -1)
#         features_output = F.dropout(features, self.dropout)
#         logits = self.regressor(features_output)

#         return self.head(logits)

#     def save(self, path: str='./model'):
#         torch.save(self.state_dict(), path)

#     def load(self, path: str, **kwargs):
#         self.load_state_dict(torch.load(path, **kwargs))
    
#     def _get_conv_output_shape(self, layer: nn.Module):
#         sample_data = torch.randn(1, CHANNELS, config.HEIGHT, config.WIDTH)
#         output = layer(sample_data).flatten()
#         return output.shape[0]


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#         super().__init__()

#         self.reduction = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1),
#             nn.BatchNorm2d(in_channels // 2),
#         )
#         self.conv1 = nn.Conv2d(in_channels // 2, out_channels, kernel_size, stride, padding)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
#             nn.BatchNorm2d(out_channels),
#         )
    
#     def forward(self, x):
#         identity = x
#         out = self.relu(self.bn1(self.conv1(self.reduction(x))))
#         out = self.bn2(self.conv2(out))
#         identity = self.downsample(identity)
#         out += identity
#         out = self.relu(out)
#         return out
    
class HandDTTR(nn.Module):
    def __init__(self):
        super(HandDTTR, self).__init__()
        
        # Darknet-53 backbone (simplified)
        self._kernel_size = 3
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, self._kernel_size, padding=1),
            nn.LeakyReLU(0.1),
            darknet_block(32, 64),
            nn.MaxPool2d(2, 2),
            darknet_block(64, 128),
            nn.MaxPool2d(2, 2),
            darknet_block(128, 256),
            nn.MaxPool2d(2, 2),
            darknet_block(256, 512),
            nn.MaxPool2d(2, 2),
            darknet_block(512, 1024),
        )
        
        # Keypoint predictor
        self.predictor = nn.Conv2d(1024, 2 * N_KEYPOINTS, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        keypoints = self.predictor(features)
        
        # Reshape the keypoints to match the target tensor's shape
        batch_size = keypoints.size(0)
        keypoints_reshaped = keypoints.view(batch_size, 2 * N_KEYPOINTS, -1).mean(dim=-1)
        
        return keypoints_reshaped
    
    def save(self, path: str='./model'):
        torch.save(self.state_dict(), path)

    def load(self, path: str, **kwargs):
        self.load_state_dict(torch.load(path, **kwargs))

def darknet_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(out_channels, in_channels, 1),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )
