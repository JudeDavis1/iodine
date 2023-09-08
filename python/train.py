import os
import torch
from torchvision import transforms

from model import Runner


LR = 0.0001
EPOCHS = 7
BATCH_SIZE = 64
MODEL_NAME = './HandDTTR.model'

device = torch.device('mps')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def main():
    print(f'[*] Using {str(device).upper()} backend')

    trainer = Runner(
        device=str(device),
        dropout=0.1,
    )
    if os.path.exists(MODEL_NAME):
        trainer.model.load(MODEL_NAME)
        print('[*] Loaded model...')
    
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Number of Parameters: {str(n_params / 1_000_000)} M")
    
    trainer.fit(
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,

        transform=transform,
        max_data=50_000,
    )
    trainer.plot_train_data()


if __name__ == '__main__':
    main()
