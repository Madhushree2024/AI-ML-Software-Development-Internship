import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ← forces file save without display popup
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import os

# ── Config ────────────────────────────────────────────────
DATA_DIR  = "C:/Users/hp/cnn_project/data"
CKPT_PATH = "C:/Users/hp/cnn_project/resnet50_cat_dog.pth"
SAVE_DIR  = "C:/Users/hp/cnn_project"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Load checkpoint ───────────────────────────────────────
ckpt        = torch.load(CKPT_PATH, map_location=DEVICE)
class_names = ckpt["class_names"]
h1          = ckpt["history_p1"]
h2          = ckpt["history_p2"]
print(f"Classes       : {class_names}")
print(f"history_p1 keys: {list(h1.keys())}")
print(f"history_p2 keys: {list(h2.keys())}")

# ── Rebuild model ─────────────────────────────────────────
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(2048, 2)
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE).eval()
print("Model loaded successfully\n")

# ── Test transform ────────────────────────────────────────
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Test DataLoader ───────────────────────────────────────
test_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"), test_transform
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0
)
print(f"Test images: {len(test_dataset)}")

# ── Run inference ─────────────────────────────────────────
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs  = inputs.to(DEVICE)
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── Accuracy ──────────────────────────────────────────────
test_acc = np.mean(all_preds == all_labels)
print(f"\n{'='*45}")
print(f"  Test Accuracy : {test_acc*100:.2f}%")
print(f"  Correct       : {int(np.sum(all_preds == all_labels))} / {len(test_dataset)}")
print(f"{'='*45}")

# ── Classification report ─────────────────────────────────
print("\nClassification Report:")
print(classification_report(all_labels, all_preds,
                            target_names=class_names))

# ── Plot 1: Confusion Matrix ──────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.5)
ax.set_title(f"Confusion Matrix — Test Acc: {test_acc*100:.2f}%")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
plt.tight_layout()
path1 = os.path.join(SAVE_DIR, "confusion_matrix.png")
fig.savefig(path1, dpi=150)
plt.close(fig)
print(f"Saved → {path1}")

# ── Plot 2: Training Curves ───────────────────────────────
train_acc  = h1["train_acc"]  + h2["train_acc"]
val_acc    = h1["val_acc"]    + h2["val_acc"]
train_loss = h1["train_loss"] + h2["train_loss"]
val_loss   = h1["val_loss"]   + h2["val_loss"]
epochs     = range(1, len(train_acc) + 1)
p1_end     = len(h1["train_acc"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_acc,  "b-o", label="Train Acc",  markersize=4)
ax1.plot(epochs, val_acc,    "r-o", label="Val Acc",    markersize=4)
ax1.axvline(x=p1_end + 0.5, color="gray",  linestyle="--", label="Phase boundary")
ax1.axhline(y=0.90,          color="green", linestyle=":",  label=">90% target")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)
ax1.set_ylim([0.5, 1.05])

ax2.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=4)
ax2.plot(epochs, val_loss,   "r-o", label="Val Loss",   markersize=4)
ax2.axvline(x=p1_end + 0.5, color="gray", linestyle="--", label="Phase boundary")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(alpha=0.3)

plt.suptitle("ResNet50 Fine-tuning — Cat vs Dog", fontsize=13)
plt.tight_layout()
path2 = os.path.join(SAVE_DIR, "training_curves.png")
fig.savefig(path2, dpi=150)
plt.close(fig)
print(f"Saved → {path2}")

# ── Plot 3: Sample Predictions ────────────────────────────
sample_indices = np.random.choice(len(test_dataset), 10, replace=False)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

for i, idx in enumerate(sample_indices):
    img_tensor, true_label = test_dataset[idx]
    img = img_tensor * std + mean
    img = img.permute(1, 2, 0).numpy().clip(0, 1)

    pred_label = all_preds[idx]
    confidence = all_probs[idx][pred_label] * 100
    correct    = pred_label == true_label

    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(
        f"True : {class_names[true_label]}\n"
        f"Pred : {class_names[pred_label]} ({confidence:.1f}%)",
        color="green" if correct else "red",
        fontsize=9
    )

plt.suptitle("Sample Predictions  (green=correct, red=wrong)", fontsize=12)
plt.tight_layout()
path3 = os.path.join(SAVE_DIR, "sample_predictions.png")
fig.savefig(path3, dpi=150)
plt.close(fig)
print(f"Saved → {path3}")

print("\nAll 3 plots saved successfully!")