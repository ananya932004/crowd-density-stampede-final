"""
Simple CSRNet-style density estimator wrapper.

This module provides a lightweight integration point for a crowd-counting CNN.
It tries to import PyTorch and will raise an informative error if torch is not installed.

It expects pretrained weights to be placed in `weights/csrnet.pth` (not committed).
"""
from pathlib import Path
import io
import base64
from PIL import Image
import numpy as np

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
CSR_WEIGHTS = WEIGHTS_DIR / "csrnet.pth"

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms
except Exception:
    torch = None


class CSRNet(nn.Module):
    def __init__(self, load_pretrained_frontend=True):
        super().__init__()
        # frontend: VGG-16 features conv layers
        vgg = torchvision.models.vgg16_bn(pretrained=load_pretrained_frontend)
        features = list(vgg.features.children())
        # keep features up to conv5
        self.frontend = nn.Sequential(*features[:33])

        # backend: dilated conv layers
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.output_layer = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


_MODEL = None


def load_model(device="cpu"):
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if torch is None:
        raise RuntimeError("PyTorch is required for the density estimator. Please install torch.")
    if not CSR_WEIGHTS.exists():
        raise FileNotFoundError(f"CSRNet weights not found at {CSR_WEIGHTS}. Please place pretrained weights there.")
    model = CSRNet(load_pretrained_frontend=False)
    state = torch.load(str(CSR_WEIGHTS), map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        # try loading partial dict
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    _MODEL = model
    return model


def weights_present():
    return CSR_WEIGHTS.exists()


def estimate_density(pil_image, device="cpu", max_side=1024):
    """Estimate a density map and total count from a PIL image.

    Returns: total_count (float), heatmap_png_bytes (bytes), density_map (2D np.array)
    """
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch to use density estimator.")

    if not CSR_WEIGHTS.exists():
        raise FileNotFoundError(f"CSRNet weights not found at {CSR_WEIGHTS}.")

    model = load_model(device=device)

    # preprocess: convert to RGB tensor, normalize like ImageNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Optionally downscale image to max_side for faster inference
    w, h = pil_image.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_proc = pil_image.resize((new_w, new_h), Image.BILINEAR)
    else:
        pil_proc = pil_image

    # ensure dims divisible by 8
    pw, ph = pil_proc.size
    new_w = (pw // 8) * 8
    new_h = (ph // 8) * 8
    if new_w == 0 or new_h == 0:
        raise ValueError("Image is too small for density estimation")
    if new_w != pw or new_h != ph:
        pil_proc = pil_proc.resize((new_w, new_h), Image.BILINEAR)

    img_t = transform(pil_proc).unsqueeze(0)  # 1,C,H,W

    with torch.no_grad():
        img_t = img_t.to(device)
        out = model(img_t)
        density = out.squeeze(0).squeeze(0).cpu().numpy()

    total_count = float(density.sum()) / (scale * scale) if scale != 1.0 else float(density.sum())

    # create a simple heatmap from density using normalization
    norm = density - density.min()
    if norm.max() > 0:
        norm = (norm / norm.max() * 255).astype('uint8')
    else:
        norm = (norm * 255).astype('uint8')

    # resize heatmap back to original image size for visualization
    heat = Image.fromarray(norm).convert('L').resize((w, h), Image.BILINEAR)

    buf = io.BytesIO()
    heat.save(buf, format='PNG')
    heat_bytes = buf.getvalue()

    return total_count, heat_bytes, density
