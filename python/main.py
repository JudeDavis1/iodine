import warnings

warnings.filterwarnings("ignore")

import json
import time

import cv2
import torch

import config
from data import N_KEYPOINTS, GaussianNormalizer
from model import Runner
from train import transform

vc = cv2.VideoCapture(0)
runner = Runner(device=torch.device("mps"))
runner.model.load("./HandDTTR.model", map_location="cpu")
runner.model.eval()

norm_: dict = json.load(open("./data/normalization.json", "r"))

y_normalizer = GaussianNormalizer(
    mean=norm_["y_mean"],
    std=norm_["y_std"],
)

runner.model = runner.model.to("mps")

WIDTH = config.WIDTH
HEIGHT = config.HEIGHT


@torch.no_grad()
def run():
    i = 0
    while True:
        _, og_frame = vc.read()
        if og_frame is None:
            print("DEBUG: FRAME IS NONE")
        og_frame = cv2.resize(og_frame, (WIDTH, HEIGHT))
        input_tensor: torch.Tensor = transform(og_frame).to("mps", non_blocking=True)

        x_coords, y_coords = runner.predict(input_tensor, y_normalizer)
        x_coords, y_coords = x_coords.astype(int), y_coords.astype(int)
        image = og_frame.copy()
        for j in range(N_KEYPOINTS):
            image = cv2.circle(
                image,
                (x_coords[j], y_coords[j]),
                radius=4,
                thickness=-1,
                color=(65, 10, 10),
            )

        image = cv2.flip(image, 1)
        cv2.imshow("winname", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        i += 1

    vc.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
