"""Load the trained GAN generator and create hand-written digit images."""
import io
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

from app.gan_model import NOISE_DIM, Generator

MODEL_PATH = Path(__file__).parent / "models" / "gan_generator.pth"


class DigitGenerator:
    def __init__(self, model_path=MODEL_PATH):
        self.model = Generator()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()  # BatchNorm uses running statistics at inference

    def generate_png(self, num_images=16, seed=None):
        """Generate num_images digits and return them as one PNG grid (bytes)."""
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        else:
            g.seed()
        z = torch.randn(num_images, NOISE_DIM, generator=g)
        with torch.no_grad():
            images = self.model(z)  # values in [-1, 1]
        images = (images + 1) / 2  # back to [0, 1]
        nrow = math.ceil(math.sqrt(num_images))
        grid = make_grid(images, nrow=nrow, padding=2, pad_value=1.0)
        # enlarge 4x so the digits are easy to see
        img = to_pil_image(grid).resize((grid.shape[2] * 4, grid.shape[1] * 4), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
