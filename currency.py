"""Indian currency note classifier (transfer learning). Shared by training and the live app."""
import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMAGE_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

ARCHS = ("mobilenet_v3_large", "efficientnet_b0", "resnet50")


def build_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "resnet50":
        m = models.resnet50(weights="DEFAULT" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch!r}, choose from {ARCHS}")
    return m


def train_transform():
    # No horizontal flips: a mirrored note never appears in real life.
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CurrencyClassifier:
    def __init__(self, path: str, device: str):
        ckpt = torch.load(path, map_location=device)
        self.classes = ckpt["classes"]
        self.device = device
        self.model = build_model(ckpt["arch"], len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.tf = eval_transform()

    @torch.no_grad()
    def predict(self, frame_bgr) -> tuple[str, float]:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        x = self.tf(image).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(dim=1)[0]
        conf, idx = probs.max(dim=0)
        return self.classes[idx.item()], conf.item()
