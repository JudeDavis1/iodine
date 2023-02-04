import warnings
warnings.filterwarnings('ignore')

import cv2
import torch
import threading
import numpy as np

from tqdm import tqdm

import config

from train import transform
from classify import HandDTTR


INTER = 2
CHANNELS = 3

vc = cv2.VideoCapture(0)
model = HandDTTR.load()

print(model.validate())

def run():
    frames = []
    classes = {1: 'Hand', 0: 'No hand'}

    t = tqdm()

    i = 0
    while True:
        _, frame = vc.read()
        frame = np.array(cv2.resize(frame, (config.WIDTH, config.HEIGHT)))
        # frames.append(list(np.array(frame).flatten()))

        i += 1

        output = model.predict(frame.reshape(1, -1))[0]
        t.set_description(classes[output])

        # cv2.imshow('winname', frame)

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
