import sys
import torch
# import nncf

from torchvision import transforms
from torch.utils.data import DataLoader

# from nncf.torch import create_compressed_model, register_default_init_args

from data import IMGDataset
from model import IodynFrameClassifier

LR = .0002
EPOCHS = 10
BATCH_SIZE = 128
N_NO_HAND = 12

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def main():
    global model
    print(f'Using {str(device).upper()} backend')

    # - Load dataset and create loader to help us split data into batches
    dataset = IMGDataset(transform=transform)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = IodineFrameClassifier().to(device)
    # config = register_default_init_args(nncf.NNCFConfig.from_dict(model.state_dict()), dl)

    # _, model = create_compressed_model(model, config)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.7, 0.999))
    criterion = torch.nn.BCELoss()

    ttl_steps = len(dl)

    for i in range(EPOCHS):
        try:
            for j, (img, labels) in enumerate(dl):
                model.zero_grad()

                outputs = model(img)
                loss = criterion(outputs.flatten().float(), labels.float())

                loss.backward()
                optimizer.step()

                print(f'Epoch: {i} Step: {j + 1}/{ttl_steps} Loss: {loss}')
        except KeyboardInterrupt:
            model.save()
            sys.exit(0)

        model.save()



if __name__ == '__main__':
    main()


