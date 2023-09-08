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
        self.model = HandDTTR(dropout=dropout)
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
        gradient_acc: int=4,
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
        total_loss = 0
        loss_val = None

        n_steps_per_batch = len(loader) // gradient_acc
        n_steps = self.epochs * (n_steps_per_batch)
        
        try:
            print('[*] Beginning training...')
            for i in (t := tqdm(range(1, self.epochs + 1))):
                for j, (img, y) in enumerate(loader):
                    cur_step = (i * len(loader)) + j

                    # get the next batch from the loader
                    y = y.float().flatten().to(self.device)
                    output: torch.Tensor = self.model(img.to(self.device)).float().flatten()
                    loss: torch.Tensor = criterion(output, y) / gradient_acc

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    total_loss += loss.mean().item()

                    if (cur_step + 1) % gradient_acc == 0 or (cur_step + 1) == n_steps:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        loss_val_str = round(loss_val.item(), 6) if loss_val else "N/A"

                        update_num = (j // gradient_acc) + 1
                        t.set_description(
                            f"Epoch {i} - Batch: {update_num}/{n_steps_per_batch} - Train loss: {total_loss:.6f}  Validation loss: {loss_val_str}"
                        )
                        total_loss = 0

                    if j % 5 == 0:
                        self.model.eval()
                        img_val, y_val = next(iter(val_loader))
                        y_val = y_val.float().flatten().to(self.device)
                        output_val: torch.Tensor = self.model(img_val.to(self.device)).float().flatten()
                        loss_val: torch.Tensor = criterion(output_val, y_val)
                        self.model.train()

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
    def __init__(self, dropout: float=0.1):
        super(HandDTTR, self).__init__()
        
        # Darknet-53 backbone (simplified)
        self._kernel_size = 3
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, self._kernel_size, padding=1),
            nn.LeakyReLU(0.1),
            darknet_block(32, 64, dropout),
            nn.MaxPool2d(2, 2),
            darknet_block(64, 128, dropout),
            nn.MaxPool2d(2, 2),
            darknet_block(128, 256, dropout),
            nn.MaxPool2d(2, 2),
            darknet_block(256, 512, dropout),
            nn.MaxPool2d(2, 2),
            darknet_block(512, 1024, dropout),
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

def darknet_block(in_channels, out_channels, dropout: float=0.1):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1),
        
        nn.Dropout(dropout),
        
        nn.Conv2d(out_channels, in_channels, 1),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(0.1),

        nn.Dropout(dropout),
        
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1),

        nn.Dropout(dropout),
    )
    return ResidualBlock(in_channels, out_channels, block)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block):
        super(ResidualBlock, self).__init__()
        self.block = block
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.block(x)
        out += self.shortcut(identity)
        return out
