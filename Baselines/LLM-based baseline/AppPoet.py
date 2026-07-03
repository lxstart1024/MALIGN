import os
import json
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


SEED = 2026
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WORK_DIR = r"D:/data/AppPoet_outputs"
FEATURE_CSV = os.path.join(WORK_DIR, "features.csv")

LLM_TEXT_JSONL = os.path.join(WORK_DIR, "deepseek_view_summaries.jsonl")
EMB_NPZ = os.path.join(WORK_DIR, "deepseek_summary_embeddings.npz")
RESULT_CSV = os.path.join(WORK_DIR, "deepseek_appoet_cv_results.csv")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
TEMPERATURE = 0.0
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.3

LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



N_SPLITS = 5
NUM_EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3
DROPOUT = 0.3


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_json_list(x):
    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    return []


def list_to_bullets(xs: List[str]) -> str:
    if not xs:
        return "None"
    return "\n".join([f"- {x}" for x in xs])


def load_features(feature_csv: str) -> pd.DataFrame:
    if not os.path.exists(feature_csv):
        raise FileNotFoundError(
            f"Cannot find feature file: {feature_csv}. "
            f"Please generate or extract AppPoet-style features first."
        )

    df = pd.read_csv(feature_csv, encoding="utf-8-sig")

    required_cols = [
        "apk_path",
        "package",
        "requested_permission",
        "used_permission",
        "restricted_api",
        "suspicious_api",
        "url",
        "uses_feature",
        "label",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in features.csv: {missing}")

    list_cols = [
        "requested_permission",
        "used_permission",
        "restricted_api",
        "suspicious_api",
        "url",
        "uses_feature",
    ]

    for col in list_cols:
        df[col] = df[col].apply(parse_json_list)

    df["label"] = df["label"].astype(int)

    print("[Loaded features]")
    print(df[["apk_path", "package", "label"]].head())
    print("\n[Label distribution]")
    print(df["label"].value_counts())

    return df



def get_deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if api_key is None or not api_key.strip():
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. "
            "Please configure it in PyCharm Run Configuration or PowerShell."
        )

    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )


def call_deepseek(prompt: str) -> str:
    client = get_deepseek_client()

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful Android malware analysis assistant. "
                            "You must only summarize facts supported by the given features."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            last_error = e
            print(f"[DeepSeek Error] attempt={attempt}/{MAX_RETRIES}: {repr(e)}")
            time.sleep(2.0 * attempt)

    raise RuntimeError(f"DeepSeek call failed after {MAX_RETRIES} retries: {repr(last_error)}")


def build_view_prompt(row: pd.Series, view_name: str) -> str:
    package = row["package"]

    if view_name == "permission":
        view_title = "Permission View"
        view_desc = (
            "This view analyzes application behavior based on requested permissions "
            "declared in AndroidManifest.xml and permissions inferred as actually used."
        )
        content = f"""
Requested permissions:
{list_to_bullets(row["requested_permission"])}

Used permissions:
{list_to_bullets(row["used_permission"])}
"""

    elif view_name == "api":
        view_title = "API View"
        view_desc = (
            "This view analyzes application behavior based on restricted APIs and suspicious APIs "
            "that may reflect access to sensitive resources or security-relevant operations."
        )
        content = f"""
Restricted APIs:
{list_to_bullets(row["restricted_api"])}

Suspicious APIs:
{list_to_bullets(row["suspicious_api"])}
"""

    elif view_name == "url_feature":
        view_title = "URL & Uses-feature View"
        view_desc = (
            "This view analyzes application behavior based on hardcoded URLs and declared "
            "hardware/software feature requirements."
        )
        content = f"""
URLs:
{list_to_bullets(row["url"])}

Uses-feature declarations:
{list_to_bullets(row["uses_feature"])}
"""

    else:
        raise ValueError(f"Unknown view_name: {view_name}")

    prompt = f"""
You are an expert in Android security and static malware analysis.

Task:
Generate a concise behavior summary for the following Android application from the {view_title}.

Package name:
{package}

View description:
{view_desc}

Extracted features:
{content}

Output requirements:
1. Summarize only based on the provided features.
2. Focus on behavior semantics and potential security risks.
3. Do not invent missing features or external facts.
4. Do not directly decide whether the app is malware.
5. Do not provide suggestions for evasion or misuse.
6. Output one concise paragraph in English.

Behavior summary:
"""
    return prompt.strip()



def load_existing_jsonl(path: str) -> Dict[str, Dict]:
    done = {}

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                done[obj["apk_path"]] = obj

    return done


def generate_deepseek_summaries(df: pd.DataFrame, output_jsonl: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    done = load_existing_jsonl(output_jsonl)

    rows = []
    fout = open(output_jsonl, "a", encoding="utf-8")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating DeepSeek summaries"):
        apk_path = row["apk_path"]

        if apk_path in done:
            rows.append(done[apk_path])
            continue

        print(f"\n[DeepSeek] Processing: {row['package']}")

        permission_summary = call_deepseek(build_view_prompt(row, "permission"))
        time.sleep(SLEEP_BETWEEN_CALLS)

        api_summary = call_deepseek(build_view_prompt(row, "api"))
        time.sleep(SLEEP_BETWEEN_CALLS)

        url_feature_summary = call_deepseek(build_view_prompt(row, "url_feature"))
        time.sleep(SLEEP_BETWEEN_CALLS)

        app_text = "\n".join([
            "[Permission View Summary]",
            permission_summary,
            "[API View Summary]",
            api_summary,
            "[URL & Uses-feature View Summary]",
            url_feature_summary,
        ])

        obj = {
            "apk_path": apk_path,
            "package": row["package"],
            "label": int(row["label"]),
            "permission_summary": permission_summary,
            "api_summary": api_summary,
            "url_feature_summary": url_feature_summary,
            "app_text": app_text,
        }

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fout.flush()

        rows.append(obj)

    fout.close()

    out_df = pd.DataFrame(rows)
    print(f"\n[Saved DeepSeek summaries] {output_jsonl}")
    return out_df


def build_or_load_embeddings(text_df: pd.DataFrame, emb_npz: str):
    if os.path.exists(emb_npz):
        data = np.load(emb_npz, allow_pickle=True)
        print(f"[Loaded embeddings] {emb_npz}")
        return data["X"], data["y"]

    texts = text_df["app_text"].fillna("").tolist()
    y = text_df["label"].astype(int).to_numpy()

    print(f"[Embedding] Loading local model: {LOCAL_EMBED_MODEL}")
    embedder = SentenceTransformer(LOCAL_EMBED_MODEL)

    X = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    X = np.asarray(X, dtype=np.float32)

    np.savez_compressed(emb_npz, X=X, y=y)
    print(f"[Saved embeddings] {emb_npz}")
    print(f"[Embedding shape] {X.shape}")

    return X, y


class MLPDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = DROPOUT):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def metrics_binary(y_true, y_pred) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def train_eval_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = MLPDetector(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    counts = np.bincount(y_train, minlength=2)
    class_weights = counts.sum() / (2.0 * np.maximum(counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss={np.mean(losses):.4f}")

    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32, device=DEVICE))
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    return metrics_binary(y_test, pred)


def run_cross_validation(X: np.ndarray, y: np.ndarray, result_csv: str) -> pd.DataFrame:
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")

        result = train_eval_one_fold(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            y[test_idx],
        )

        result["fold"] = fold
        rows.append(result)

        print(
            f"Fold={fold}, "
            f"Acc={result['acc']:.4f}, "
            f"Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"MCC={result['mcc']:.4f}"
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    print("\n===== Cross-validation Summary =====")
    print("\n[Mean]")
    print(result_df.mean(numeric_only=True))

    print("\n[Std]")
    print(result_df.std(numeric_only=True))

    print(f"\n[Saved results] {result_csv}")

    return result_df



def main():
    set_seed(SEED)
    os.makedirs(WORK_DIR, exist_ok=True)

    print(f"[Device] {DEVICE}")
    print(f"[Feature CSV] {FEATURE_CSV}")
    print(f"[DeepSeek model] {DEEPSEEK_MODEL}")
    print(f"[Embedding model] {LOCAL_EMBED_MODEL}")

    feature_df = load_features(FEATURE_CSV)

    text_df = generate_deepseek_summaries(
        df=feature_df,
        output_jsonl=LLM_TEXT_JSONL,
    )

    X, y = build_or_load_embeddings(
        text_df=text_df,
        emb_npz=EMB_NPZ,
    )

    run_cross_validation(
        X=X,
        y=y,
        result_csv=RESULT_CSV,
    )


if __name__ == "__main__":
    main()