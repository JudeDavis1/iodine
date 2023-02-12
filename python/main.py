import warnings
warnings.filterwarnings('ignore')

import cv2
import torch
import numpy as np

from tqdm import tqdm

import config

from train import transform
from model import Runner
from data import N_KEYPOINTS


INTER = 2

vc = cv2.VideoCapture(0)
runner = Runner()
runner.model.load("./HandDTTR.model")
runner.model.eval()

WIDTH = config.WIDTH
HEIGHT = config.HEIGHT

def run():
    i = 0
    while True:
        _, og_frame = vc.read()
        if og_frame is None: print("DEBUG: FRAME IS NONE")
        og_frame = cv2.resize(og_frame, (WIDTH, HEIGHT))
        frame = transform(og_frame).numpy()
        input_tensor = torch.tensor(frame)

        i += 1

        # og_frame = np.transpose(og_frame, (1, 2, 0))
        x_coords, y_coords = runner.predict(input_tensor)
        x_coords, y_coords = x_coords.astype(int), y_coords.astype(int)
        image = og_frame.copy()
        for i in range(N_KEYPOINTS):
            image = cv2.circle(image, (x_coords[i], y_coords[i]), radius=4, thickness=-1, color=(65, 10, 10))
        
        image = cv2.flip(image, 1)
        
        cv2.imshow('winname', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
