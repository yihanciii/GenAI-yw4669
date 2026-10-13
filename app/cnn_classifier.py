"""Load the trained CNN and classify a single image."""
import io
from pathlib import Path

import torch
from PIL import Image

from app.cnn_model import CLASSES, AssignmentCNN, transform

MODEL_PATH = Path(__file__).parent / "models" / "cnn_cifar10.pth"


class CNNClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.model = AssignmentCNN()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()  # inference mode

    def predict(self, image_bytes, top_k=3):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = transform(image).unsqueeze(0)  # shape: 1 x 3 x 64 x 64
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        top = torch.topk(probs, k=top_k)
        return {
            "predicted_class": CLASSES[top.indices[0].item()],
            "confidence": round(top.values[0].item(), 4),
            "top_k": [
                {"class": CLASSES[i.item()], "probability": round(p.item(), 4)}
                for p, i in zip(top.values, top.indices)
            ],
        }
