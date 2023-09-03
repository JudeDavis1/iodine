import warnings
warnings.filterwarnings('ignore')

import cv2
import time
import torch

import config

from train import transform
from model import Runner
from data import N_KEYPOINTS


vc = cv2.VideoCapture(0)
runner = Runner(device=torch.device('mps'))
runner.model.load("./HandDTTR.model", map_location='cpu')
runner.model.eval()

runner.model = runner.model.to('mps')

WIDTH = config.WIDTH
HEIGHT = config.HEIGHT

@torch.no_grad()
def run():
    i = 0
    while True:
        _, og_frame = vc.read()
        if og_frame is None: print("DEBUG: FRAME IS NONE")
        og_frame = cv2.resize(og_frame, (WIDTH, HEIGHT))
        input_tensor: torch.Tensor = transform(og_frame)
        
        x_coords, y_coords = runner.predict(input_tensor.to('mps', non_blocking=True))
        x_coords, y_coords = x_coords.astype(int), y_coords.astype(int)
        image = og_frame.copy()
        for j in range(N_KEYPOINTS):
            image = cv2.circle(image, (x_coords[j], y_coords[j]), radius=4, thickness=-1, color=(65, 10, 10))
        
        image = cv2.flip(image, 1)
        time.sleep(0.01)
        cv2.imshow('winname', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    
    vc.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
