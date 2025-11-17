import base64
import io
from typing import Dict, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastmcp import FastMCP
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms
from uniface import RetinaFace

from config import data_config
from utils.helpers import get_model

mcp = FastMCP("gaze-mcp")

# --------------------------------------
# Load Model (same logic as your API)
# --------------------------------------
DATASET = "gaze360"
MODEL_NAME = "resnet34"
WEIGHT_PATH = f"weights/{MODEL_NAME}.pt"

dataset_config = data_config[DATASET]
BINS = dataset_config["bins"]
BINWIDTH = dataset_config["binwidth"]
ANGLE = dataset_config["angle"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idx_tensor = torch.arange(BINS, device=device, dtype=torch.float32)

face_detector = RetinaFace()

gaze_detector = get_model(MODEL_NAME, BINS, inference_mode=True)
state_dict = torch.load(WEIGHT_PATH, map_location=device)
gaze_detector.load_state_dict(state_dict)
gaze_detector.to(device)
gaze_detector.eval()


# --------------------------------------
# Preprocess
# --------------------------------------
def pre_process(image: NDArray[np.uint8]) -> torch.Tensor:
    """Preprocesses an image by resizing it and normalizing it."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transform(image).unsqueeze(0)


# --------------------------------------
# MCP Tool: image → yaw/pitch
# Input must be base64-encoded image
# --------------------------------------
@mcp.tool()
def estimate_front_facing_gazes(image_b64: str) -> Dict[str, Union[int, bool]]:
    """
    Provide yaw and pitch for each detected face.

    Args:
        image_b64: Image as a base64 string.

    Returns:

    """
    img_bytes = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame = cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

    all_faces_towards_cam = True
    faces = face_detector.detect(frame)

    with torch.no_grad():
        for face in faces:
            x1, y1, x2, y2 = map(int, face["bbox"][:4])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            batch = pre_process(crop).to(device)

            pitch, yaw = gaze_detector(batch)

            pitch_sm = F.softmax(pitch, dim=1)
            yaw_sm = F.softmax(yaw, dim=1)

            pitch_pred = torch.sum(pitch_sm * idx_tensor, dim=1) * BINWIDTH - ANGLE
            yaw_pred = torch.sum(yaw_sm * idx_tensor, dim=1) * BINWIDTH - ANGLE

            pitch_pred = pitch_pred.cpu().item()
            yaw_pred = yaw_pred.cpu().item()

            print(pitch_pred, yaw_pred)

            # The criteria for
            is_facing_cam = pitch_pred**2 + yaw_pred**2 <= 10**2

            if not is_facing_cam:
                all_faces_towards_cam = False
                break

    return {"num_faces": len(faces), "all_faces_towards_cam": all_faces_towards_cam}


if __name__ == "__main__":
    mcp.run()
