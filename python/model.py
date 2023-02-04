import warnings
warnings.filterwarnings('ignore')

import torch

from torch import nn
from torch.nn import functional as F

# - Constants
CHANNELS = 3


class IodynFrameClassifier(nn.Module):

    def __init__(self, n_batches=5):
        super().__init__()

        self._featuremap = 64


        self.feature_extractor = nn.Sequential(
            nn.Conv2d(CHANNELS, self._featuremap, 4, bias=False),
            nn.BatchNorm2d(self._featuremap),
            nn.MaxPool2d(2, 2),
            # nn.ReLU(),
            # nn.Conv2d(self._featuremap, 4, 4, bias=False),
            # nn.BatchNorm2d(4),
            # nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            FlattenExcludeBatchDim(),
            nn.LazyLinear(128),
            nn.Linear(128, 1),
        )

        self.dropout = nn.Dropout(.4)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2).float()
        features = self.feature_extractor(x)
        preds = self.dropout(self.classifier(features))
        logits = F.softmax(preds)

        print(logits)

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
