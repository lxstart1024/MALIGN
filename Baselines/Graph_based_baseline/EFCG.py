import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef
except ImportError as exc:
    raise ImportError("Please install scikit-learn: pip install scikit-learn") from exc


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class AppGraph:
    path: str
    label: int
    calls: List[Tuple[str, str]]
    sequence: List[str]

def infer_label_from_path(path: str) -> int:
    lower = path.lower()
    parts = lower.replace("\\", "/").split("/")
    if "benign" in parts:
        return 0
    if "malware" in parts or "malicious" in parts:
        return 1
    raise ValueError(
        f"Cannot infer label from path: {path}. "
        "Use folder names 'benign' and 'malware', or include a 'label' field in JSON."
    )


def load_app_graphs(root_dir: str) -> List[AppGraph]:
    apps: List[AppGraph] = []

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if not name.lower().endswith(".json"):
                continue

            path = os.path.join(dirpath, name)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            label = int(obj.get("label", infer_label_from_path(path)))

            raw_calls = obj.get("calls", [])
            calls: List[Tuple[str, str]] = []
            for e in raw_calls:
                if not isinstance(e, (list, tuple)) or len(e) != 2:
                    continue
                caller, callee = str(e[0]), str(e[1])
                calls.append((caller, callee))

            if "sequence" in obj and isinstance(obj["sequence"], list):
                sequence = [str(x) for x in obj["sequence"]]
            else:
                sequence = []
                for caller, callee in calls:
                    sequence.append(caller)
                    sequence.append(callee)

            if len(sequence) == 0 and len(calls) == 0:
                continue

            apps.append(AppGraph(path=path, label=label, calls=calls, sequence=sequence))

    if not apps:
        raise RuntimeError(f"No valid JSON app graphs found under {root_dir}")

    return apps


class FunctionVocab:
    def __init__(self, min_count: int = 1):
        self.min_count = min_count
        self.func2id: Dict[str, int] = {"<UNK>": 0}
        self.id2func: List[str] = ["<UNK>"]

    def build(self, sequences: List[List[str]]) -> None:
        counter = Counter()
        for seq in sequences:
            counter.update(seq)

        for func, cnt in counter.items():
            if cnt >= self.min_count and func not in self.func2id:
                self.func2id[func] = len(self.id2func)
                self.id2func.append(func)

    def encode(self, func: str) -> int:
        return self.func2id.get(func, 0)

    def __len__(self) -> int:
        return len(self.id2func)


class CBOWDataset(Dataset):
    def __init__(self, sequences: List[List[int]], window_size: int = 2):
        self.samples: List[Tuple[List[int], int]] = []
        self.window_size = window_size

        for seq in sequences:
            if len(seq) < 2 * window_size + 1:
                continue
            for i in range(window_size, len(seq) - window_size):
                context = seq[i - window_size:i] + seq[i + 1:i + window_size + 1]
                target = seq[i]
                self.samples.append((context, target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        context, target = self.samples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class FunctionCBOW(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(context_ids)
        hidden = emb.mean(dim=1)
        logits = self.output(hidden)
        return logits


def train_function_embeddings(
    apps: List[AppGraph],
    vocab: FunctionVocab,
    embed_dim: int = 100,
    window_size: int = 2,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    encoded_sequences = [[vocab.encode(f) for f in app.sequence] for app in apps]
    cbow_data = CBOWDataset(encoded_sequences, window_size=window_size)

    model = FunctionCBOW(len(vocab), embed_dim=embed_dim).to(device)

    if len(cbow_data) == 0:
        print("[FunctionEmbedding] Not enough sequences for CBOW training. Using randomly initialized embeddings.")
        return model.embedding.weight.detach().cpu()

    loader = DataLoader(cbow_data, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        losses = []
        for context, target in loader:
            context = context.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(context)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        print(f"[FunctionEmbedding] Epoch {epoch + 1}/{epochs}, Loss={np.mean(losses):.4f}")

    return model.embedding.weight.detach().cpu()


@dataclass
class EFCGSample:
    x: torch.Tensor
    adj: torch.Tensor
    label: int
    path: str


class EFCGDataset(Dataset):
    def __init__(
        self,
        apps: List[AppGraph],
        vocab: FunctionVocab,
        embedding_matrix: torch.Tensor,
        max_nodes: int = 800,
    ):
        self.samples: List[EFCGSample] = []
        self.vocab = vocab
        self.embedding_matrix = embedding_matrix.float()
        self.max_nodes = max_nodes

        for app in apps:
            sample = self._build_sample(app)
            if sample is not None:
                self.samples.append(sample)

        if not self.samples:
            raise RuntimeError("No E-FCG samples were constructed.")

    def _build_sample(self, app: AppGraph) -> Optional[EFCGSample]:
        funcs = set(app.sequence)
        for caller, callee in app.calls:
            funcs.add(caller)
            funcs.add(callee)

        ordered_funcs = []
        seen = set()
        for f in app.sequence:
            if f not in seen:
                ordered_funcs.append(f)
                seen.add(f)
        for caller, callee in app.calls:
            for f in (caller, callee):
                if f not in seen:
                    ordered_funcs.append(f)
                    seen.add(f)

        ordered_funcs = ordered_funcs[:self.max_nodes]
        if len(ordered_funcs) == 0:
            return None

        local_id = {f: i for i, f in enumerate(ordered_funcs)}
        n = len(ordered_funcs)

        adj = torch.zeros((n, n), dtype=torch.float32)
        for caller, callee in app.calls:
            if caller in local_id and callee in local_id:
                i = local_id[caller]
                j = local_id[callee]
                # Directed edge: caller invokes callee.
                adj[i, j] = 1.0

        node_ids = torch.tensor([self.vocab.encode(f) for f in ordered_funcs], dtype=torch.long)
        x = self.embedding_matrix[node_ids]

        return EFCGSample(x=x, adj=adj, label=int(app.label), path=app.path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EFCGSample:
        return self.samples[idx]


def efcg_collate(batch: List[EFCGSample]):
    return batch


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n = adj.size(0)
        device = adj.device

        # A_tilde = A + I.
        a_tilde = adj + torch.eye(n, device=device)

        deg = a_tilde.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
        norm = deg_inv_sqrt.unsqueeze(1) * a_tilde * deg_inv_sqrt.unsqueeze(0)

        h = norm @ x
        h = self.linear(h)
        return h


class EFCG_BLFE(nn.Module):
    def __init__(
        self,
        in_dim: int = 100,
        hidden_dims: Tuple[int, int, int] = (100, 100, 60),
        classifier_hidden: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.gcn1 = GraphConv(in_dim, hidden_dims[0])
        self.gcn2 = GraphConv(hidden_dims[0], hidden_dims[1])
        self.gcn3 = GraphConv(hidden_dims[1], hidden_dims[2])

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[2], classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 2),
        )

    def extract_feature(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, adj))
        h = self.dropout(h)
        h = F.relu(self.gcn2(h, adj))
        h = self.dropout(h)
        h = F.relu(self.gcn3(h, adj))

        # ReadOut layer: sum each feature column over nodes.
        graph_feature = h.sum(dim=0)
        return graph_feature

    def forward_one(self, sample: EFCGSample, device: torch.device) -> torch.Tensor:
        x = sample.x.to(device)
        adj = sample.adj.to(device)
        graph_feature = self.extract_feature(x, adj)
        logits = self.classifier(graph_feature)
        return logits

    def forward_batch(self, batch: List[EFCGSample], device: torch.device) -> torch.Tensor:
        logits = [self.forward_one(sample, device).unsqueeze(0) for sample in batch]
        return torch.cat(logits, dim=0)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "mcc": mcc}


def train_one_fold(
    dataset: EFCGDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
) -> Dict[str, float]:
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, collate_fn=efcg_collate
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, collate_fn=efcg_collate
    )

    model = EFCG_BLFE(dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    labels_train = [dataset.samples[i].label for i in train_idx]
    counts = np.bincount(np.array(labels_train), minlength=2)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            y = torch.tensor([s.label for s in batch], dtype=torch.long, device=device)
            optimizer.zero_grad()
            logits = model.forward_batch(batch, device)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        print(f"[Train] Epoch {epoch + 1}/{epochs}, Loss={np.mean(losses):.4f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model.forward_batch(batch, device)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            true = [s.label for s in batch]
            y_true.extend(true)
            y_pred.extend(pred)

    return compute_metrics(y_true, y_pred)


def run_cross_validation(
    dataset: EFCGDataset,
    device: torch.device,
    folds: int = 5,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.5,
) -> List[Dict[str, float]]:
    labels = np.array([s.label for s in dataset.samples], dtype=np.int64)
    indices = np.arange(len(dataset))

    min_count = np.min(np.bincount(labels, minlength=2))
    if min_count < folds:
        raise ValueError(f"The smallest class has {min_count} samples, but folds={folds}.")

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n===== Fold {fold + 1}/{folds} =====")
        metrics = train_one_fold(
            dataset=dataset,
            train_idx=train_idx,
            test_idx=test_idx,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            dropout=dropout,
        )
        print(
            f"[Test] Acc={metrics['acc']:.4f}, Prec={metrics['prec']:.4f}, "
            f"Rec={metrics['rec']:.4f}, F1={metrics['f1']:.4f}, MCC={metrics['mcc']:.4f}"
        )
        results.append(metrics)

    print("\n===== Cross-validation Summary =====")
    for key in ["acc", "prec", "rec", "f1", "mcc"]:
        vals = np.array([r[key] for r in results], dtype=np.float64)
        print(f"{key}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    return results

def extract_behavior_features(
    dataset: EFCGDataset,
    out_path: str,
    device: torch.device,
    dropout: float = 0.5,
) -> None:

    model = EFCG_BLFE(dropout=dropout).to(device)
    model.eval()

    features = []
    labels = []
    paths = []

    with torch.no_grad():
        for sample in dataset.samples:
            x = sample.x.to(device)
            adj = sample.adj.to(device)
            feat = model.extract_feature(x, adj).cpu().numpy()
            features.append(feat)
            labels.append(sample.label)
            paths.append(sample.path)

    np.save(out_path, np.asarray(features, dtype=np.float32))
    np.save(out_path.replace(".npy", "_labels.npy"), np.asarray(labels, dtype=np.int64))

    with open(out_path.replace(".npy", "_paths.txt"), "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")

    print(f"[FeatureExtraction] Saved features to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="EFCG reproduction code")

    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing app FCG JSON files.")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds. The paper uses 5-fold CV.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for the GCN classifier.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the GCN classifier.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")

    parser.add_argument("--embed_dim", type=int, default=100, help="Function embedding dimension. The paper uses 100.")
    parser.add_argument("--embed_epochs", type=int, default=5, help="Epochs for CBOW function embedding.")
    parser.add_argument("--embed_batch_size", type=int, default=512, help="Batch size for CBOW function embedding.")
    parser.add_argument("--window_size", type=int, default=2, help="CBOW context window size k. Context size is 2k.")
    parser.add_argument("--min_count", type=int, default=1, help="Minimum count for functions in vocabulary.")
    parser.add_argument("--max_nodes", type=int, default=800, help="Maximum number of nodes per E-FCG to control memory.")

    parser.add_argument("--extract_features", action="store_true", help="Only extract behavior-level features.")
    parser.add_argument("--feature_out", type=str, default="efcg_features.npy", help="Output path for extracted features.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    apps = load_app_graphs(args.root_dir)
    labels = np.array([a.label for a in apps], dtype=np.int64)
    print(f"[Data] Apps={len(apps)}, Benign={(labels == 0).sum()}, Malware={(labels == 1).sum()}")

    vocab = FunctionVocab(min_count=args.min_count)
    vocab.build([app.sequence for app in apps])
    print(f"[Vocab] Functions={len(vocab)}")

    embedding_matrix = train_function_embeddings(
        apps=apps,
        vocab=vocab,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        epochs=args.embed_epochs,
        batch_size=args.embed_batch_size,
        lr=1e-3,
        device=device,
    )

    dataset = EFCGDataset(
        apps=apps,
        vocab=vocab,
        embedding_matrix=embedding_matrix,
        max_nodes=args.max_nodes,
    )
    print(f"[EFCG] Samples={len(dataset)}")

    if args.extract_features:
        extract_behavior_features(dataset, args.feature_out, device=device, dropout=args.dropout)
        return

    run_cross_validation(
        dataset=dataset,
        device=device,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
    )

