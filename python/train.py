import os
import torch
from torchvision import transforms

from model import Runner



LR = 0.0006
EPOCHS = 200
BATCH_SIZE = 64
MODEL_NAME = './HandDTTR.model'

device = torch.device('mps')
transform = transforms.Compose([
    transforms.ToTensor(),
])

def main():
    print(f'[*] Using {str(device).upper()} backend')

    trainer = Runner(
        device=str(device),
    )
    if os.path.exists(MODEL_NAME):
        trainer.model.load(MODEL_NAME)
        print('[*] Loaded model...')
    
    trainer.fit(
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        transform=transform,
        max_data=5000
    )



if __name__ == '__main__':
    main()


