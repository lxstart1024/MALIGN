import os
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_binary_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8)

def bytes_to_markov_image(byte_arr: np.ndarray) -> np.ndarray:
    if byte_arr.dtype != np.uint8:
        byte_arr = byte_arr.astype(np.uint8)

    if len(byte_arr) < 2:
        return np.zeros((256, 256), dtype=np.float32)

    src = byte_arr[:-1].astype(np.int32)
    dst = byte_arr[1:].astype(np.int32)

    idx = src * 256 + dst
    bc = np.bincount(idx, minlength=256 * 256)
    freq = bc.reshape(256, 256).astype(np.float64)

    row_sum = freq.sum(axis=1, keepdims=True)
    prob = np.zeros((256, 256), dtype=np.float32)
    nonzero = (row_sum.squeeze(1) > 0)
    prob[nonzero] = (freq[nonzero] / row_sum[nonzero]).astype(np.float32)
    return prob

class MarkovImageFolder(Dataset):
    def __init__(self, root: str, class_to_idx: Dict[str, int] = None):
        super().__init__()
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Split folder not found: {root}")

        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        if not classes:
            raise ValueError(f"No class folders found under: {root}")

        if class_to_idx is None:
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.samples: List[Tuple[str, int]] = []
        exts = (".bin", ".dex", ".bytes")

        for c in classes:
            if c not in self.class_to_idx:
                continue
            label = self.class_to_idx[c]
            cdir = os.path.join(root, c)
            for fn in os.listdir(cdir):
                if fn.lower().endswith(exts):
                    self.samples.append((os.path.join(cdir, fn), label))

        if not self.samples:
            raise ValueError(f"No binary files found under: {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        byte_arr = read_binary_file(path)
        img = bytes_to_markov_image(byte_arr)
        x = torch.from_numpy(img).unsqueeze(0)
        return x, torch.tensor(y, dtype=torch.long)

def make_vgg_block(in_ch: int, out_ch: int, num_convs: int) -> nn.Sequential:
    layers = []
    ch = in_ch
    for _ in range(num_convs):
        layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        ch = out_ch
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class MDMCNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(1, 64, 2),
            make_vgg_block(64, 128, 2),
            make_vgg_block(128, 256, 3),
            make_vgg_block(256, 512, 3),
            make_vgg_block(512, 512, 3),
        )

        feat_dim = 512 * 8 * 8
        self.fc1 = nn.Linear(feat_dim, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc_out(x)

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), y) * bs
        n += bs

    return total_loss / n, total_acc / n

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

@dataclass
class Config:
    data_root: str
    epochs: int = 250
    batch_size: int = 32
    lr: float = 1e-3
    decay: float = 1e-6
    num_workers: int = 2
    seed: int = 42
    save_path: str = "mdmc_best.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root containing train/val/test folders.")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--decay", type=float, default=1e-6)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", type=str, default="mdmc_best.pt")
    args = ap.parse_args()

    cfg = Config(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        decay=args.decay,
        num_workers=args.num_workers,
        seed=args.seed,
        save_path=args.save_path,
    )

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    train_dir = os.path.join(cfg.data_root, "train")
    val_dir = os.path.join(cfg.data_root, "val")
    test_dir = os.path.join(cfg.data_root, "test")

    train_ds_tmp = MarkovImageFolder(train_dir, class_to_idx=None)
    class_to_idx = train_ds_tmp.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    train_ds = MarkovImageFolder(train_dir, class_to_idx=class_to_idx)
    val_ds = MarkovImageFolder(val_dir, class_to_idx=class_to_idx)
    test_ds = MarkovImageFolder(test_dir, class_to_idx=class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    print(f"[INFO] classes={class_to_idx} | train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    model = MDMCNet(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = eval_model(model, val_loader, device)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "cfg": cfg.__dict__,
            }, cfg.save_path)

        print(f"[E{epoch:03d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.4f} | best={best_val_acc:.4f}@E{best_epoch}")

    ckpt = torch.load(cfg.save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc = eval_model(model, test_loader, device)

    print(f"[TEST] loss={te_loss:.4f} acc={te_acc:.4f}")
    print(f"[INFO] best model saved: {os.path.abspath(cfg.save_path)}")
    print(f"[INFO] idx_to_class: {idx_to_class}")
