import os
import math
import random
import numpy as np
import pandas as pd
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


SRC_DIM = 768
BIN_DIM = 1536
CLASS_NAMES = ["Benign", "Adware", "Banking", "Riskware"]
CLASS2ID = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_class_from_path(root_dir: str, csv_path: str) -> int:
    rel = os.path.relpath(csv_path, root_dir)
    cls = rel.split(os.sep)[0].strip().lower()
    return CLASS2ID[cls]


def _norm_col(x) -> str:
    return str(x).replace("\ufeff", "").strip().lower()


class AppDatasetCIC2020(Dataset):
    def __init__(self, root_dir: str, csv_name: str = "vectors.csv"):
        self.root_dir = root_dir
        self.csv_name = csv_name
        self.app_paths: List[str] = []
        self.bin_labels: List[int] = []
        self.fam_labels: List[int] = []

        for dirpath, _, filenames in os.walk(root_dir):
            if csv_name in filenames:
                p = os.path.join(dirpath, csv_name)
                fam = infer_class_from_path(root_dir, p)
                self.app_paths.append(p)
                self.fam_labels.append(fam)
                self.bin_labels.append(0 if fam == CLASS2ID["benign"] else 1)

        if not self.app_paths:
            raise RuntimeError(f"No {csv_name} found under {root_dir}")

    def __len__(self):
        return len(self.app_paths)

    def __getitem__(self, idx: int) -> Dict:
        csv_path = self.app_paths[idx]
        df = pd.read_csv(csv_path, header=0, low_memory=False)

        bin_feats = df.iloc[:, 2:2 + BIN_DIM].astype(np.float32).to_numpy()
        src_feats = df.iloc[:, 2 + BIN_DIM:2 + BIN_DIM + SRC_DIM].astype(np.float32).to_numpy()

        bin_feats = np.nan_to_num(bin_feats, nan=0.0, posinf=1e3, neginf=-1e3)
        src_feats = np.nan_to_num(src_feats, nan=0.0, posinf=1e3, neginf=-1e3)

        cols_lower = [_norm_col(c) for c in df.columns]
        conf_idx = cols_lower.index("confidence")
        q_vals = df.iloc[:, conf_idx].astype(np.float32).to_numpy()

        return {
            "bin_feats": torch.from_numpy(bin_feats),
            "src_feats": torch.from_numpy(src_feats),
            "q_vals": torch.from_numpy(q_vals),
            "bin_label": int(self.bin_labels[idx]),
            "fam_label": int(self.fam_labels[idx]),
        }


def app_collate(batch):
    bin_list = [b["bin_feats"] for b in batch]
    src_list = [b["src_feats"] for b in batch]
    q_list = [b["q_vals"] for b in batch]
    bin_y = torch.tensor([b["bin_label"] for b in batch], dtype=torch.long)
    fam_y = torch.tensor([b["fam_label"] for b in batch], dtype=torch.long)
    return bin_list, src_list, q_list, bin_y, fam_y


def confusion_matrix(y_true: List[int], y_pred: List[int], n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_cm_binary(cm: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(1, cm.sum())
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    denom = math.sqrt(denom) if denom > 0 else 0.0
    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mcc": mcc}


def metrics_multiclass_macro(cm: np.ndarray) -> Dict[str, float]:
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    prec_c = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec_c = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_c = np.divide(2 * prec_c * rec_c, prec_c + rec_c, out=np.zeros_like(tp), where=(prec_c + rec_c) > 0)

    acc = tp.sum() / max(1, cm.sum())
    prec = float(np.mean(prec_c))
    rec = float(np.mean(rec_c))
    f1 = float(np.mean(f1_c))

    c = float(np.trace(cm))
    s = float(cm.sum())
    p_k = cm.sum(axis=0).astype(np.float64)
    t_k = cm.sum(axis=1).astype(np.float64)
    sum_pk_tk = float(np.sum(p_k * t_k))
    denom1 = s * s - float(np.sum(p_k * p_k))
    denom2 = s * s - float(np.sum(t_k * t_k))
    denom = math.sqrt(denom1 * denom2) if denom1 > 0 and denom2 > 0 else 0.0
    mcc = ((c * s - sum_pk_tk) / denom) if denom > 0 else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mcc": mcc}


class MalwareModelMTL(nn.Module):
    def __init__(self, proj_dim=256, app_dim=256, temperature=0.07, p_dropout=0.1, n_fam=4):
        super().__init__()
        self.temperature = temperature

        self.src_proj = nn.Sequential(
            nn.Linear(SRC_DIM, proj_dim), nn.ReLU(), nn.Dropout(p_dropout), nn.LayerNorm(proj_dim)
        )
        self.bin_proj = nn.Sequential(
            nn.Linear(BIN_DIM, proj_dim), nn.ReLU(), nn.Dropout(p_dropout), nn.LayerNorm(proj_dim)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(2 * proj_dim, 2 * proj_dim), nn.ReLU(), nn.Dropout(p_dropout), nn.Linear(2 * proj_dim, 2 * proj_dim)
        )
        self.attn_V = nn.Linear(2 * proj_dim, 128)
        self.attn_w = nn.Linear(128, 1)
        self.app_proj = nn.Sequential(
            nn.Linear(2 * proj_dim, app_dim), nn.ReLU(), nn.Dropout(p_dropout)
        )
        self.bin_head = nn.Sequential(
            nn.Linear(app_dim, app_dim), nn.ReLU(), nn.Dropout(p_dropout), nn.Linear(app_dim, 2)
        )
        self.fam_head = nn.Sequential(
            nn.Linear(app_dim, app_dim), nn.ReLU(), nn.Dropout(p_dropout), nn.Linear(app_dim, n_fam)
        )

    def encode_methods(self, bin_feats, src_feats):
        return self.src_proj(src_feats), self.bin_proj(bin_feats)

    def fuse_methods(self, z_src, z_bin):
        u = torch.cat([z_src, z_bin], dim=-1)
        g = torch.sigmoid(self.gate_net(u))
        return g * u, g

    def aggregate_app(self, fused):
        h = torch.tanh(self.attn_V(fused))
        logits = self.attn_w(h).squeeze(-1)
        alpha = torch.softmax(logits, dim=0)
        z_weighted = torch.sum(alpha.unsqueeze(-1) * fused, dim=0)
        return self.app_proj(z_weighted)

    def contrastive_loss_weighted(self, z_src_all, z_bin_all, q_all):
        z_src_n = F.normalize(z_src_all, dim=-1)
        z_bin_n = F.normalize(z_bin_all, dim=-1)
        logits = (z_src_n @ z_bin_n.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_src = F.cross_entropy(logits, labels, reduction="none")
        loss_bin = F.cross_entropy(logits.t(), labels, reduction="none")

        q_all = torch.clamp(q_all, min=0.0)
        w = q_all / (q_all.mean() + 1e-8)

        return 0.5 * ((w * loss_src).mean() + (w * loss_bin).mean())

    def forward_app(self, bin_feats, src_feats):
        z_src, z_bin = self.encode_methods(bin_feats, src_feats)
        fused, g = self.fuse_methods(z_src, z_bin)
        z_app = self.aggregate_app(fused)
        return self.bin_head(z_app), self.fam_head(z_app), g


def run_cross_validation_mtl(
    root_dir: str,
    num_folds: int = 10,
    num_epochs: int = 5,
    app_batch_size: int = 6,
    clip_weight: float = 0.05,
    lambda_sparse: float = 1e-4,
    fam_weight: float = 1.0,
    bin_weight: float = 1.0,
    top_k_methods: int = 300,
    p_dropout: float = 0.1,
    lr: float = 5e-4,
):
    set_seed(0)

    dataset = AppDatasetCIC2020(root_dir)
    n_apps = len(dataset)
    indices = np.arange(n_apps)
    np.random.shuffle(indices)

    num_folds = min(num_folds, n_apps)
    fold_sizes = np.full(num_folds, n_apps // num_folds, dtype=int)
    fold_sizes[: n_apps % num_folds] += 1

    bin_counts = np.bincount(np.array(dataset.bin_labels), minlength=2)
    fam_counts = np.bincount(np.array(dataset.fam_labels), minlength=4)

    w_bin = bin_counts.sum() / (2.0 * np.maximum(1, bin_counts))
    w_fam = fam_counts.sum() / (4.0 * np.maximum(1, fam_counts))

    w_bin_t = torch.tensor(w_bin, dtype=torch.float32, device=DEVICE)
    w_fam_t = torch.tensor(w_fam, dtype=torch.float32, device=DEVICE)

    fold_results = []
    current = 0

    for fold in range(num_folds):
        start, end = current, current + fold_sizes[fold]
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        current = end

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=app_batch_size,
            shuffle=True,
            collate_fn=app_collate,
            num_workers=2,
            persistent_workers=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=app_batch_size,
            shuffle=False,
            collate_fn=app_collate,
            num_workers=2,
            persistent_workers=True
        )

        model = MalwareModelMTL(p_dropout=p_dropout, n_fam=4).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)

        for _ in range(num_epochs):
            model.train()
            for bin_list, src_list, q_list, bin_y, fam_y in train_loader:
                optimizer.zero_grad()

                bin_y = bin_y.to(DEVICE)
                fam_y = fam_y.to(DEVICE)

                z_src_all, z_bin_all, q_all = [], [], []
                logits_bin_batch, logits_fam_batch = [], []
                gates_batch = []

                for bin_feats, src_feats, q_vals in zip(bin_list, src_list, q_list):
                    if top_k_methods and bin_feats.size(0) > top_k_methods:
                        _, topk_idx = torch.topk(q_vals, top_k_methods)
                        bin_feats = bin_feats[topk_idx]
                        src_feats = src_feats[topk_idx]
                        q_vals = q_vals[topk_idx]

                    bin_feats = bin_feats.to(DEVICE)
                    src_feats = src_feats.to(DEVICE)
                    q_vals = q_vals.to(DEVICE)

                    z_src, z_bin = model.encode_methods(bin_feats, src_feats)
                    z_src_all.append(z_src)
                    z_bin_all.append(z_bin)
                    q_all.append(q_vals)

                    fused, g = model.fuse_methods(z_src, z_bin)
                    z_app = model.aggregate_app(fused)

                    logits_bin_batch.append(model.bin_head(z_app).unsqueeze(0))
                    logits_fam_batch.append(model.fam_head(z_app).unsqueeze(0))
                    gates_batch.append(g.abs().mean())

                z_src_all = torch.cat(z_src_all, dim=0)
                z_bin_all = torch.cat(z_bin_all, dim=0)
                q_all = torch.cat(q_all, dim=0)

                logits_bin_batch = torch.cat(logits_bin_batch, dim=0)
                logits_fam_batch = torch.cat(logits_fam_batch, dim=0)

                loss_clip = model.contrastive_loss_weighted(z_src_all, z_bin_all, q_all)
                loss_bin = F.cross_entropy(logits_bin_batch, bin_y, weight=w_bin_t)
                loss_fam = F.cross_entropy(logits_fam_batch, fam_y, weight=w_fam_t)
                loss_sparse = lambda_sparse * torch.stack(gates_batch).mean()

                loss = clip_weight * loss_clip + bin_weight * loss_bin + fam_weight * loss_fam + loss_sparse
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            scheduler.step()

        model.eval()
        bin_true, bin_pred, fam_true, fam_pred = [], [], [], []
        with torch.no_grad():
            for bin_list, src_list, q_list, bin_y, fam_y in val_loader:
                for bin_feats, src_feats, q_vals in zip(bin_list, src_list, q_list):
                    if top_k_methods and bin_feats.size(0) > top_k_methods:
                        _, topk_idx = torch.topk(q_vals, top_k_methods)
                        bin_feats = bin_feats[topk_idx]
                        src_feats = src_feats[topk_idx]

                    bin_feats = bin_feats.to(DEVICE)
                    src_feats = src_feats.to(DEVICE)

                    logit_bin, logit_fam, _ = model.forward_app(bin_feats, src_feats)
                    bin_pred.append(int(torch.argmax(logit_bin).item()))
                    fam_pred.append(int(torch.argmax(logit_fam).item()))

                bin_true.extend(bin_y.tolist())
                fam_true.extend(fam_y.tolist())

        cm_bin = confusion_matrix(bin_true, bin_pred, 2)
        cm_fam = confusion_matrix(fam_true, fam_pred, 4)
        m_bin = metrics_from_cm_binary(cm_bin)
        m_fam = metrics_multiclass_macro(cm_fam)

        fold_results.append({
            "bin_acc": m_bin["acc"], "bin_prec": m_bin["prec"], "bin_rec": m_bin["rec"], "bin_f1": m_bin["f1"], "bin_mcc": m_bin["mcc"],
            "fam_acc": m_fam["acc"], "fam_prec": m_fam["prec"], "fam_rec": m_fam["rec"], "fam_f1": m_fam["f1"], "fam_mcc": m_fam["mcc"],
        })

    return fold_results
