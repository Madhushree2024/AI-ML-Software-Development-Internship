# fix_save.py
import torch
import torch.nn as nn
from torchvision import models

CKPT_PATH = "C:/Users/hp/cnn_project/resnet50_cat_dog.pth"
DEVICE    = torch.device("cpu")

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
print("Keys in checkpoint:", ckpt.keys())