import os, shutil, random
from pathlib import Path

def split_dataset(source_dir, output_dir, split=(0.70, 0.15, 0.15), seed=42):
    random.seed(seed)
    splits = {"train": split[0], "val": split[1], "test": split[2]}

    for class_name in os.listdir(source_dir):
        class_path = Path(source_dir) / class_name
        if not class_path.is_dir():
            continue

        images = list(class_path.glob("*.jpg")) + \
                 list(class_path.glob("*.jpeg")) + \
                 list(class_path.glob("*.png"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * splits["train"])
        n_val   = int(n * splits["val"])

        split_images = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:]
        }

        for split_name, files in split_images.items():
            dest = Path(output_dir) / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)

        print(f"{class_name}: {n_train} train | {n_val} val | {len(split_images['test'])} test")

# ── Check image counts before splitting ──────────────────
source = "C:/Users/hp/cnn_project/animals"
for cls in ["cat", "dog"]:
    imgs = list(Path(source, cls).glob("*.jpg")) + \
           list(Path(source, cls).glob("*.jpeg")) + \
           list(Path(source, cls).glob("*.png"))
    print(f"{cls}: {len(imgs)} images found")

# ── Run split ─────────────────────────────────────────────
split_dataset(
    source_dir="C:/Users/hp/cnn_project/animals",
    output_dir="C:/Users/hp/cnn_project/data"
)

print("\nDone! Folder structure created at C:/Users/hp/cnn_project/data/")