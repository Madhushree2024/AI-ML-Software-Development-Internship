import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# ── Config ────────────────────────────────────────────────
DATA_DIR    = "C:/Users/hp/cnn_project/data"
NUM_CLASSES = 2
BATCH_SIZE  = 32
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Step 2: Transforms (already done — included here) ─────
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}

# ── DataLoaders ───────────────────────────────────────────
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ["train", "val"]
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                  shuffle=(x == "train"), num_workers=0)
    for x in ["train", "val"]
}
class_names = image_datasets["train"].classes
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

print(f"Classes : {class_names}")
print(f"Train   : {dataset_sizes['train']} images")
print(f"Val     : {dataset_sizes['val']} images")

# ── Step 3: Load ResNet50 + Replace Final Layer ───────────
def build_model(freeze_backbone=True):
    # Load pretrained ResNet50 (ImageNet weights)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    print(f"\nResNet50 loaded with ImageNet pretrained weights")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("Backbone frozen — only FC head will train in Phase 1")

    # ResNet50 original FC: Linear(2048 → 1000)
    # Replace with: Dropout → Linear(2048 → 2)
    in_features = model.fc.in_features          # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )
    print(f"FC head replaced: Linear({in_features} → {NUM_CLASSES})")
    print(f"Output classes  : {class_names}")

    model = model.to(DEVICE)
    return model

model = build_model(freeze_backbone=True)

# ── Verify only FC head is trainable ─────────────────────
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable params : {trainable:,} / {total:,}")
print(f"Frozen params    : {total - trainable:,}")

# ── Quick sanity check: one forward pass ─────────────────
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f"\nForward pass OK — output shape: {out.shape}")  # expect [1, 2]
print(f"Raw logits: {out}")

#---------------------------

import copy
import torch.optim as optim

# ── Training Loop ─────────────────────────────────────────
def train_model(model, optimizer, scheduler, num_epochs, phase_name):
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0
    best_wts  = copy.deepcopy(model.state_dict())
    history   = {"train_loss": [], "val_loss": [],
                 "train_acc": [],  "val_acc": []}

    print(f"\n{'='*50}")
    print(f"  {phase_name}")
    print(f"{'='*50}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}  {'─'*30}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss     = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train" and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"  {phase:5s} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}"
                  f"  ({int(epoch_acc * dataset_sizes[phase])}/{dataset_sizes[phase]})")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                print(f"  ✅ New best val accuracy: {best_acc:.4f} — weights saved")

    print(f"\n🏆 Best Val Accuracy ({phase_name}): {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model, history


# ── Phase 1: Train FC Head Only (backbone frozen) ─────────
model = build_model(freeze_backbone=True)

optimizer_phase1 = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
scheduler_phase1 = optim.lr_scheduler.StepLR(
    optimizer_phase1, step_size=2, gamma=0.5
)

model, history_p1 = train_model(
    model,
    optimizer_phase1,
    scheduler_phase1,
    num_epochs=5,
    phase_name="Phase 1 — Training classifier head (backbone frozen)"
)


# ── Phase 2: Unfreeze All — Full Fine-tune ────────────────
print("\n\nUnfreezing full network for Phase 2...")
for param in model.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params now: {trainable:,}")

# Differential LRs: backbone gets 10x smaller LR than head
optimizer_phase2 = optim.SGD([
    {"params": [p for name, p in model.named_parameters()
                if "fc" not in name],  "lr": 1e-4},   # backbone
    {"params": model.fc.parameters(), "lr": 1e-3},    # head
], momentum=0.9, weight_decay=1e-4)

scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_phase2, T_max=15
)

model, history_p2 = train_model(
    model,
    optimizer_phase2,
    scheduler_phase2,
    num_epochs=15,
    phase_name="Phase 2 — Full network fine-tuning"
)


# ── Save Best Model ───────────────────────────────────────
save_path = "C:/Users/hp/cnn_project/resnet50_cat_dog.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names":      class_names,
    "history_p1":       history_p1,
    "history_p2":       history_p2,
}, save_path)
print(f"\n💾 Model saved → {save_path}")


# ── Plot Training Curves ──────────────────────────────────
import matplotlib.pyplot as plt

def plot_history(h1, h2):
    # Combine both phases
    train_acc  = h1["train_acc"]  + h2["train_acc"]
    val_acc    = h1["val_acc"]    + h2["val_acc"]
    train_loss = h1["train_loss"] + h2["train_loss"]
    val_loss   = h1["val_loss"]   + h2["val_loss"]
    epochs     = range(1, len(train_acc) + 1)
    p1_end     = len(h1["train_acc"])          # mark phase boundary

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, train_acc, "b-o", label="Train Acc", markersize=4)
    ax1.plot(epochs, val_acc,   "r-o", label="Val Acc",   markersize=4)
    ax1.axvline(x=p1_end + 0.5, color="gray", linestyle="--", label="Phase boundary")
    ax1.axhline(y=0.90, color="green", linestyle=":", label=">90% target")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=4)
    ax2.plot(epochs, val_loss,   "r-o", label="Val Loss",   markersize=4)
    ax2.axvline(x=p1_end + 0.5, color="gray", linestyle="--", label="Phase boundary")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("ResNet50 Fine-tuning — Cat vs Dog", fontsize=13)
    plt.tight_layout()
    plt.savefig("C:/Users/hp/cnn_project/training_curves.png", dpi=150)
    plt.show()
    print("📊 Training curves saved → training_curves.png")

plot_history(history_p1, history_p2)