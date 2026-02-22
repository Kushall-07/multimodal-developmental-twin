import json
import os
import time
import random
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score

# Use project-local cache so training works in restricted envs
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[3] / "models" / "torch_cache"))

from .config import BATCH_SIZE, NUM_WORKERS, MODEL_PATH, META_PATH, MODEL_VERSION, TARGET, MAP_TO_TARGET
from .config import DATA_DIR
from .dataset import get_transforms


class FilteredFER(Dataset):
    """FER2013 filtered to 4 classes (angry, happy, sad, bored) with optional remap (neutral -> bored)."""
    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {c: i for i, c in enumerate(TARGET)}

        for fer_class_dir in self.root.iterdir():
            if not fer_class_dir.is_dir():
                continue
            fer_name = fer_class_dir.name.lower().strip()
            if fer_name not in MAP_TO_TARGET:
                continue
            target_name = MAP_TO_TARGET[fer_name]
            y = self.class_to_idx[target_name]
            for img_path in fer_class_dir.glob("*.jpg"):
                self.samples.append((img_path, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


def balanced_subset(ds, per_class: int, seed: int = 42):
    """Stratified subset: sample evenly from each class. Works with FilteredFER (.samples) or ImageFolder (.targets)."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    if hasattr(ds, "samples"):
        for idx in range(len(ds)):
            y = ds.samples[idx][1]
            by_class[y].append(idx)
    else:
        for idx, y in enumerate(ds.targets):
            by_class[y].append(idx)

    picked = []
    for y, idxs in by_class.items():
        rng.shuffle(idxs)
        picked.extend(idxs[:per_class])
    rng.shuffle(picked)
    return Subset(ds, picked)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"
    train_tfms = get_transforms(train=True)
    test_tfms = get_transforms(train=False)

    train_ds_full = FilteredFER(str(train_dir), transform=train_tfms)
    test_ds_full = FilteredFER(str(test_dir), transform=test_tfms)
    num_classes = 4

    train_ds = balanced_subset(train_ds_full, per_class=1500)
    test_ds = balanced_subset(test_ds_full, per_class=500)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # Class weights: length 4 (angry, happy, sad, bored)
    counts = Counter([y for _, y in train_ds_full.samples])
    class_counts = [counts.get(i, 0) for i in range(4)]
    class_weights = np.array([1.0 / max(c, 1) for c in class_counts], dtype=np.float32)
    class_weights = class_weights / class_weights.mean()
    print("class_counts:", dict(counts))
    print("class_weights:", class_weights.tolist())
    assert len(class_weights) == 4, f"Expected 4-class weights, got {len(class_weights)}"

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device), label_smoothing=0.05)

    t0 = time.time()

    # Stage A: freeze backbone, train classifier head
    for p in model.features.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    warmup_epochs = 5  # Stage A: train head only

    for ep in range(1, warmup_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        # quick eval
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(yb.numpy().tolist())

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        print(f"stage=A epoch={ep} test_acc={acc:.4f} macro_f1={f1_macro:.4f} weighted_f1={f1_weighted:.4f}")

    # Stage B: unfreeze last few blocks and fine-tune with lower LR
    for p in model.features[-4:].parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    finetune_epochs = 8  # Stage B: fine-tune last 4 blocks

    for ep in range(1, finetune_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(yb.numpy().tolist())

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        print(f"stage=B epoch={ep} test_acc={acc:.4f} macro_f1={f1_macro:.4f} weighted_f1={f1_weighted:.4f}")

    idx_to_class = {v: k for k, v in train_ds_full.class_to_idx.items()}
    torch.save({
        "model_state": model.state_dict(),
        "class_to_idx": train_ds_full.class_to_idx,
        "idx_to_class": idx_to_class,
        "model_version": MODEL_VERSION,
    }, MODEL_PATH)

    meta = {
        "classes": TARGET,
        "mapping_note": "bored is a proxy mapped from FER2013 neutral",
        "model_version": MODEL_VERSION,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "finished_at_utc": time.time(),
        "device": str(device),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print("✅ Emotion model training done")
    print("Saved model to:", MODEL_PATH)
    print("Saved metadata to:", META_PATH)


if __name__ == "__main__":
    train()
