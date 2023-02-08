import warnings
warnings.filterwarnings('ignore')

import cv2
import torch
import threading
import numpy as np

from tqdm import tqdm

import config

from train import transform
from model import HandDTTR


INTER = 2
CHANNELS = 3

vc = cv2.VideoCapture(1)
model = HandDTTR()
model.load("./HandDTTR.model")
model.eval()

def run():
    frames = []
    classes = {1: 'Hand', 0: 'No hand'}

    t = tqdm()

    i = 0
    while True:
        _, frame = vc.read()
        if frame is None: print("FRAME IS NONE")
        frame = transform(cv2.resize(frame, (config.WIDTH, config.HEIGHT))).numpy()
        input_tensor = torch.tensor([frame])

        i += 1

        output = model(input_tensor)
        prediction = output
        t.set_description(str(prediction))
        cv2.imshow('winname', np.transpose(frame, (1, 2, 0)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Flatten a multidimensional list recursively
def flatten(x: list) -> list:
    # Account for single lists
    if len(x) == 1:
        if type(x[0]) == list:
            result = flatten(x[0])
        else:
            # Array already flattened
            result = x
    elif type(x[0]) == list:
        # Concat first flattened item to the other flattened items
        result = flatten(x[0]) + flatten(x[1:])
    else:
        # If the first item is a scalar, work on the rest of the list
        # The rest of the list will be taken care of by the above code
        result = x[0] + flatten(x[1:])

    return result


if __name__ == '__main__':
    run()

    vc.release()
    cv2.destroyAllWindows()
