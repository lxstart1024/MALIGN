import os
import random
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

SRC_DIM = 768
BIN_DIM = 1536

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BENIGN_NAME = "benign"

FAMILY_NAMES = ["Adware", "Trojan", "Riskware", "Others"]
FAMILY2ID = {name.lower(): i for i, name in enumerate(FAMILY_NAMES)}

IGNORE_FAMILY_LABEL = -100

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def infer_labels_from_path(root_dir: str, csv_path: str) -> Tuple[int, int]:
    rel = os.path.relpath(csv_path, root_dir)
    parts = rel.split(os.sep)

    if len(parts) < 2:
        raise ValueError(f"[LabelInfer] Path too shallow: {csv_path}")

    cls = parts[0].strip().lower()

    if cls == BENIGN_NAME:
        return 0, IGNORE_FAMILY_LABEL

    if cls not in FAMILY2ID:
        raise ValueError(
            f"[LabelInfer] Unknown class folder '{parts[0]}' in: {csv_path}. "
            f"Expected 'Benign' or one of {FAMILY_NAMES}."
        )

    return 1, FAMILY2ID[cls]

class AppDatasetMTL(Dataset):

    def __init__(self, root_dir: str, csv_name: str = "vectors.csv"):
        self.root_dir = root_dir
        self.csv_name = csv_name

        self.app_paths: List[str] = []
        self.bin_labels: List[int] = []
        self.fam_labels: List[int] = []

        for dirpath, _, filenames in os.walk(root_dir):
            if csv_name in filenames:
                csv_path = os.path.join(dirpath, csv_name)

                bin_label, fam_label = infer_labels_from_path(root_dir, csv_path)

                self.app_paths.append(csv_path)
                self.bin_labels.append(bin_label)
                self.fam_labels.append(fam_label)

        if not self.app_paths:
            raise RuntimeError(f"[AppDatasetMTL] No {csv_name} found under {root_dir}")

        self._print_stats()

    def _print_stats(self) -> None:
        bin_counts = np.bincount(np.array(self.bin_labels), minlength=2)

        fam_valid = [y for y in self.fam_labels if y != IGNORE_FAMILY_LABEL]

        if fam_valid:
            fam_counts = np.bincount(
                np.array(fam_valid),
                minlength=len(FAMILY_NAMES)
            )
        else:
            fam_counts = np.zeros(len(FAMILY_NAMES), dtype=int)

        print(f"[AppDatasetMTL] Apps: {len(self.app_paths)}")
        print(f"  - Benign:  {bin_counts[0]}")
        print(f"  - Malware: {bin_counts[1]}")

        for i, name in enumerate(FAMILY_NAMES):
            print(f"  - {name}: {fam_counts[i]}")

    def __len__(self) -> int:
        return len(self.app_paths)

    def __getitem__(self, idx: int) -> Dict:
        csv_path = self.app_paths[idx]

        bin_label = int(self.bin_labels[idx])
        fam_label = int(self.fam_labels[idx])

        df = pd.read_csv(csv_path, header=0, low_memory=False)

        expected_min_cols = 2 + BIN_DIM + SRC_DIM

        if df.shape[1] < expected_min_cols:
            raise ValueError(
                f"[AppDatasetMTL] CSV columns insufficient: {csv_path}; "
                f"cols={df.shape[1]} < expected_min_cols={expected_min_cols}"
            )

        bin_feats = (
            df.iloc[:, 2:2 + BIN_DIM]
            .astype(np.float32)
            .to_numpy()
        )

        src_feats = (
            df.iloc[:, 2 + BIN_DIM:2 + BIN_DIM + SRC_DIM]
            .astype(np.float32)
            .to_numpy()
        )

        bin_feats = np.nan_to_num(
            bin_feats,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3
        )

        src_feats = np.nan_to_num(
            src_feats,
            nan=0.0,
            posinf=1e3,
            neginf=-1e3
        )

        cols_lower = [
            str(c).replace("\ufeff", "").strip().lower()
            for c in df.columns
        ]

        if "confidence" in cols_lower:
            conf_idx = cols_lower.index("confidence")
            q_vals = df.iloc[:, conf_idx].astype(np.float32).to_numpy()
            q_vals = np.nan_to_num(
                q_vals,
                nan=1.0,
                posinf=1.0,
                neginf=0.0
            )
        else:
            q_vals = np.ones(df.shape[0], dtype=np.float32)

        return {
            "bin_feats": torch.from_numpy(bin_feats),
            "src_feats": torch.from_numpy(src_feats),
            "q_vals": torch.from_numpy(q_vals),
            "bin_label": bin_label,
            "fam_label": fam_label,
            "app_path": csv_path,
        }


def app_collate(batch):
    bin_list = [b["bin_feats"] for b in batch]
    src_list = [b["src_feats"] for b in batch]
    q_list = [b["q_vals"] for b in batch]

    bin_y = torch.tensor(
        [b["bin_label"] for b in batch],
        dtype=torch.long
    )

    fam_y = torch.tensor(
        [b["fam_label"] for b in batch],
        dtype=torch.long
    )

    paths = [b["app_path"] for b in batch]

    return bin_list, src_list, q_list, bin_y, fam_y, paths


def confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    n_classes: int
) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        t = int(t)
        p = int(p)

        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    return cm


def metrics_from_cm_binary(cm: np.ndarray) -> Dict[str, float]:
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]

    acc = (tp + tn) / max(1, cm.sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec) / max(1e-12, prec + rec)

    denom_val = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    denom = math.sqrt(denom_val) if denom_val > 0 else 0.0

    mcc = ((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "mcc": mcc,
    }


def metrics_multiclass_macro(cm: np.ndarray) -> Dict[str, float]:
    if cm.sum() == 0:
        return {
            "acc": 0.0,
            "prec": 0.0,
            "rec": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
        }

    tp = np.diag(cm).astype(np.float64)

    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    prec_c = np.divide(
        tp,
        tp + fp,
        out=np.zeros_like(tp),
        where=(tp + fp) > 0
    )

    rec_c = np.divide(
        tp,
        tp + fn,
        out=np.zeros_like(tp),
        where=(tp + fn) > 0
    )

    f1_c = np.divide(
        2 * prec_c * rec_c,
        prec_c + rec_c,
        out=np.zeros_like(tp),
        where=(prec_c + rec_c) > 0
    )

    acc = tp.sum() / max(1, cm.sum())
    prec = float(np.mean(prec_c))
    rec = float(np.mean(rec_c))
    f1 = float(np.mean(f1_c))

    # Multi-class MCC
    c = float(np.trace(cm))
    s = float(cm.sum())

    p_k = cm.sum(axis=0).astype(np.float64)
    t_k = cm.sum(axis=1).astype(np.float64)

    sum_pk_tk = float(np.sum(p_k * t_k))

    denom1 = s * s - float(np.sum(p_k * p_k))
    denom2 = s * s - float(np.sum(t_k * t_k))

    denom = math.sqrt(denom1 * denom2) if denom1 > 0 and denom2 > 0 else 0.0

    mcc = ((c * s - sum_pk_tk) / denom) if denom > 0 else 0.0

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "mcc": mcc,
    }


class MalwareModelMTL(nn.Module):
    def __init__(
        self,
        proj_dim: int = 256,
        app_dim: int = 256,
        temperature: float = 0.07,
        p_dropout: float = 0.1,
        n_fam: int = 4
    ):
        super().__init__()

        self.temperature = temperature
        self.n_fam = n_fam

        self.src_proj = nn.Sequential(
            nn.Linear(SRC_DIM, proj_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.LayerNorm(proj_dim),
        )

        self.bin_proj = nn.Sequential(
            nn.Linear(BIN_DIM, proj_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.LayerNorm(proj_dim),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(2 * proj_dim, 2 * proj_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(2 * proj_dim, 2 * proj_dim),
        )

        self.attn_V = nn.Linear(2 * proj_dim, 128)
        self.attn_w = nn.Linear(128, 1)

        self.app_proj = nn.Sequential(
            nn.Linear(2 * proj_dim, app_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
        )

        self.bin_head = nn.Sequential(
            nn.Linear(app_dim, app_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(app_dim, 2),
        )

        self.fam_head = nn.Sequential(
            nn.Linear(app_dim, app_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(app_dim, n_fam),
        )

    def encode_methods(self, bin_feats, src_feats):
        z_bin = self.bin_proj(bin_feats)
        z_src = self.src_proj(src_feats)

        return z_src, z_bin

    def fuse_methods(self, z_src, z_bin):
        u = torch.cat([z_src, z_bin], dim=-1)

        g = torch.sigmoid(self.gate_net(u))
        fused = g * u

        return fused, g

    def aggregate_app(self, fused):
        h = torch.tanh(self.attn_V(fused))
        logits = self.attn_w(h).squeeze(-1)

        alpha = torch.softmax(logits, dim=0)

        z_weighted = torch.sum(alpha.unsqueeze(-1) * fused, dim=0)
        z_app = self.app_proj(z_weighted)

        return z_app, alpha

    def contrastive_loss_weighted(self, z_src_all, z_bin_all, q_all):
        if z_src_all.size(0) < 2:
            return torch.tensor(0.0, device=z_src_all.device)

        z_src_n = F.normalize(z_src_all, dim=-1)
        z_bin_n = F.normalize(z_bin_all, dim=-1)

        logits = (z_src_n @ z_bin_n.t()) / self.temperature

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_src_vec = F.cross_entropy(
            logits,
            labels,
            reduction="none"
        )

        loss_bin_vec = F.cross_entropy(
            logits.t(),
            labels,
            reduction="none"
        )

        q_all = torch.clamp(q_all, min=0.0)

        mean_q = q_all.mean()

        if mean_q <= 0:
            w = torch.ones_like(q_all)
        else:
            w = q_all / (mean_q + 1e-8)

        loss = 0.5 * (
            (w * loss_src_vec).mean()
            + (w * loss_bin_vec).mean()
        )

        return loss

    def forward_app(self, bin_feats, src_feats):
        z_src, z_bin = self.encode_methods(bin_feats, src_feats)

        fused, g = self.fuse_methods(z_src, z_bin)

        z_app, alpha = self.aggregate_app(fused)

        logit_bin = self.bin_head(z_app)
        logit_fam = self.fam_head(z_app)

        return logit_bin, logit_fam, g, alpha


def compute_fold_class_weights(
    dataset: AppDatasetMTL,
    train_idx: np.ndarray
):
    train_bin_labels = np.array(
        [dataset.bin_labels[i] for i in train_idx],
        dtype=np.int64
    )

    train_fam_labels = np.array(
        [dataset.fam_labels[i] for i in train_idx],
        dtype=np.int64
    )

    bin_counts = np.bincount(train_bin_labels, minlength=2)
    w_bin = bin_counts.sum() / (2.0 * np.maximum(1, bin_counts))

    mal_fam_labels = train_fam_labels[train_bin_labels == 1]

    if len(mal_fam_labels) == 0:
        w_fam = np.ones(len(FAMILY_NAMES), dtype=np.float32)
    else:
        fam_counts = np.bincount(
            mal_fam_labels,
            minlength=len(FAMILY_NAMES)
        )

        w_fam = fam_counts.sum() / (
            len(FAMILY_NAMES) * np.maximum(1, fam_counts)
        )

    w_bin_t = torch.tensor(
        w_bin,
        dtype=torch.float32,
        device=DEVICE
    )

    w_fam_t = torch.tensor(
        w_fam,
        dtype=torch.float32,
        device=DEVICE
    )

    return w_bin_t, w_fam_t


def build_stratification_labels(dataset: AppDatasetMTL) -> np.ndarray:

    strat_labels = []

    for bin_y, fam_y in zip(dataset.bin_labels, dataset.fam_labels):
        if bin_y == 0:
            strat_labels.append(0)
        else:
            if fam_y == IGNORE_FAMILY_LABEL:
                raise ValueError(
                    "[Stratification] Malicious sample has invalid family label."
                )

            strat_labels.append(int(fam_y) + 1)

    return np.array(strat_labels, dtype=np.int64)

def run_cross_validation_mtl(
    root_dir: str = r"D:\data",
    num_folds: int = 10,
    num_epochs: int = 10,
    app_batch_size: int = 50,

    clip_weight: float = 0.05,
    fam_weight: float = 1.0,
    bin_weight: float = 1.0,

    top_k_methods: int = 300,

    p_dropout: float = 0.1,
    lr: float = 5e-4,

    num_workers: int = 2,
):
    set_seed(0)

    print(f"[Device] {DEVICE}")

    dataset = AppDatasetMTL(root_dir=root_dir)
    n_apps = len(dataset)

    indices = np.arange(n_apps)
    strat_labels = build_stratification_labels(dataset)

    min_class_count = np.min(np.bincount(strat_labels))

    if min_class_count < num_folds:
        raise ValueError(
            f"[CrossValidation] The smallest class has only {min_class_count} samples, "
            f"which is smaller than num_folds={num_folds}. "
            f"Please reduce num_folds or add more samples."
        )

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=0
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, strat_labels)):
        print(f"\n===== Fold {fold + 1}/{num_folds} =====")

        w_bin_t, w_fam_t = compute_fold_class_weights(dataset, train_idx)

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=app_batch_size,
            shuffle=True,
            collate_fn=app_collate,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=app_batch_size,
            shuffle=False,
            collate_fn=app_collate,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

        model = MalwareModelMTL(
            p_dropout=p_dropout,
            n_fam=len(FAMILY_NAMES)
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=2,
            gamma=0.7
        )

        for epoch in range(num_epochs):
            model.train()

            epoch_total = []
            epoch_clip = []
            epoch_bin = []
            epoch_fam = []

            for bin_list, src_list, q_list, bin_y, fam_y, _paths in train_loader:
                optimizer.zero_grad()

                bin_y = bin_y.to(DEVICE)
                fam_y = fam_y.to(DEVICE)

                z_src_all = []
                z_bin_all = []
                q_all = []

                logits_bin_batch = []
                logits_fam_batch = []

                for bin_feats, src_feats, q_vals in zip(bin_list, src_list, q_list):
                    if (
                        top_k_methods is not None
                        and top_k_methods > 0
                        and bin_feats.size(0) > top_k_methods
                    ):
                        _, topk_idx = torch.topk(q_vals, top_k_methods)

                        bin_feats = bin_feats[topk_idx]
                        src_feats = src_feats[topk_idx]
                        q_vals = q_vals[topk_idx]

                    bin_feats = bin_feats.to(DEVICE)
                    src_feats = src_feats.to(DEVICE)
                    q_vals = q_vals.to(DEVICE)

                    z_src, z_bin = model.encode_methods(
                        bin_feats,
                        src_feats
                    )

                    z_src_all.append(z_src)
                    z_bin_all.append(z_bin)
                    q_all.append(q_vals)

                    fused, _g = model.fuse_methods(z_src, z_bin)
                    z_app, _alpha = model.aggregate_app(fused)

                    logits_bin_batch.append(
                        model.bin_head(z_app).unsqueeze(0)
                    )

                    logits_fam_batch.append(
                        model.fam_head(z_app).unsqueeze(0)
                    )

                z_src_all = torch.cat(z_src_all, dim=0)
                z_bin_all = torch.cat(z_bin_all, dim=0)
                q_all = torch.cat(q_all, dim=0)

                logits_bin_batch = torch.cat(logits_bin_batch, dim=0)
                logits_fam_batch = torch.cat(logits_fam_batch, dim=0)

                loss_clip = model.contrastive_loss_weighted(
                    z_src_all,
                    z_bin_all,
                    q_all
                )

                loss_bin = F.cross_entropy(
                    logits_bin_batch,
                    bin_y,
                    weight=w_bin_t
                )

                malicious_mask = (bin_y == 1)

                if malicious_mask.any():
                    loss_fam = F.cross_entropy(
                        logits_fam_batch[malicious_mask],
                        fam_y[malicious_mask],
                        weight=w_fam_t
                    )
                else:
                    loss_fam = torch.tensor(0.0, device=DEVICE)

                total = (
                    clip_weight * loss_clip
                    + bin_weight * loss_bin
                    + fam_weight * loss_fam
                )

                total.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

                epoch_total.append(float(total.item()))
                epoch_clip.append(float(loss_clip.item()))
                epoch_bin.append(float(loss_bin.item()))
                epoch_fam.append(float(loss_fam.item()))

            scheduler.step()

            print(
                f"[Train] Fold={fold + 1}, Epoch={epoch + 1}/{num_epochs}, "
                f"Loss={np.mean(epoch_total):.4f}, "
                f"CLIP={np.mean(epoch_clip):.4f}, "
                f"BIN={np.mean(epoch_bin):.4f}, "
                f"FAM={np.mean(epoch_fam):.4f}, "
                f"LR={scheduler.get_last_lr()[0]:.6f}"
            )

        model.eval()

        bin_true = []
        bin_pred = []

        fam_true = []
        fam_pred = []

        with torch.no_grad():
            for bin_list, src_list, q_list, bin_y, fam_y, _paths in val_loader:
                batch_bin_y = bin_y.tolist()
                batch_fam_y = fam_y.tolist()

                for app_i, (bin_feats, src_feats, q_vals) in enumerate(
                    zip(bin_list, src_list, q_list)
                ):
                    if (
                        top_k_methods is not None
                        and top_k_methods > 0
                        and bin_feats.size(0) > top_k_methods
                    ):
                        _, topk_idx = torch.topk(q_vals, top_k_methods)

                        bin_feats = bin_feats[topk_idx]
                        src_feats = src_feats[topk_idx]

                    bin_feats = bin_feats.to(DEVICE)
                    src_feats = src_feats.to(DEVICE)

                    logit_bin, logit_fam, _g, _alpha = model.forward_app(
                        bin_feats,
                        src_feats
                    )

                    pred_bin = int(torch.argmax(logit_bin).item())
                    pred_fam = int(torch.argmax(logit_fam).item())

                    true_bin = int(batch_bin_y[app_i])
                    true_fam = int(batch_fam_y[app_i])

                    bin_true.append(true_bin)
                    bin_pred.append(pred_bin)

                    if true_bin == 1 and true_fam != IGNORE_FAMILY_LABEL:
                        fam_true.append(true_fam)
                        fam_pred.append(pred_fam)

        cm_bin = confusion_matrix(
            bin_true,
            bin_pred,
            n_classes=2
        )

        m_bin = metrics_from_cm_binary(cm_bin)

        cm_fam = confusion_matrix(
            fam_true,
            fam_pred,
            n_classes=len(FAMILY_NAMES)
        )

        m_fam = metrics_multiclass_macro(cm_fam)

        print(
            f"[Val-BIN] Acc={m_bin['acc']:.4f}, "
            f"Prec={m_bin['prec']:.4f}, "
            f"Rec={m_bin['rec']:.4f}, "
            f"F1={m_bin['f1']:.4f}, "
            f"MCC={m_bin['mcc']:.4f}"
        )

        print(
            f"[Val-FAM] Acc={m_fam['acc']:.4f}, "
            f"Prec(Macro)={m_fam['prec']:.4f}, "
            f"Rec(Macro)={m_fam['rec']:.4f}, "
            f"F1(Macro)={m_fam['f1']:.4f}, "
            f"MCC={m_fam['mcc']:.4f}"
        )

        print("[Val-BIN Confusion Matrix]")
        print(cm_bin)

        print("[Val-FAM Confusion Matrix]")
        print(cm_fam)

        fold_results.append({
            "bin_acc": m_bin["acc"],
            "bin_prec": m_bin["prec"],
            "bin_rec": m_bin["rec"],
            "bin_f1": m_bin["f1"],
            "bin_mcc": m_bin["mcc"],

            "fam_acc": m_fam["acc"],
            "fam_prec": m_fam["prec"],
            "fam_rec": m_fam["rec"],
            "fam_f1": m_fam["f1"],
            "fam_mcc": m_fam["mcc"],
        })

    print("\n===== K-Fold Summary =====")

    df = pd.DataFrame(fold_results)

    print("\n[Mean]")
    print(df.mean(numeric_only=True))

    print("\n[Std]")
    print(df.std(numeric_only=True))

    return fold_results


if __name__ == "__main__":
    run_cross_validation_mtl(
        root_dir=r"D:\data",
        num_folds=10,
        num_epochs=10,
        app_batch_size=50,

        top_k_methods=300,

        clip_weight=0.05,
        bin_weight=1.0,
        fam_weight=1.0,

        p_dropout=0.1,
        lr=5e-4,

        num_workers=2,
    )