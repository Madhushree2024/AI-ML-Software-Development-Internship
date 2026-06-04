import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys, os

# ── Config ────────────────────────────────────────────────
CKPT_PATH = "C:/Users/hp/cnn_project/resnet50_cat_dog.pth"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────
ckpt        = torch.load(CKPT_PATH, map_location=DEVICE)
class_names = ckpt["class_names"]

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(2048, 2)
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()

# ── Transform ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Predict function ──────────────────────────────────────
def predict(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found → {image_path}")
        return

    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]

    cat_conf = probs[class_names.index("cat")].item() * 100
    dog_conf = probs[class_names.index("dog")].item() * 100

    pred  = class_names[torch.argmax(probs).item()]
    conf  = max(cat_conf, dog_conf)

    print(f"\nImage     : {os.path.basename(image_path)}")
    print(f"Prediction: {pred.upper()}")
    print(f"Confidence: {conf:.2f}%")
    print(f"  cat → {cat_conf:.2f}%")
    print(f"  dog → {dog_conf:.2f}%")
    return pred, conf

# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default test: pick one image from test set
        test_dir = "C:/Users/hp/cnn_project/data/test"
        sample   = None
        for cls in os.listdir(test_dir):
            cls_path = os.path.join(test_dir, cls)
            for f in os.listdir(cls_path):
                if f.endswith((".jpg", ".jpeg", ".png")):
                    sample = os.path.join(cls_path, f)
                    break
            if sample:
                break
        if sample:
            predict(sample)
        else:
            print("No test images found.")
    else:
        # Predict on image passed as argument
        predict(sys.argv[1])