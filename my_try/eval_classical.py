"""
非-LLM 经典 baseline（Step 3 巩固）

给 LLM baseline 建立参照下限。所有基线都在 CPU 上完成，全部跑完 <30s。

包含 5 个 baseline：
  B1 Majority      - 全预测 acceptable（多数类）
  B2 Random        - 按训练集先验随机
  B3 LengthThresh  - 以句子 token 数 > 阈值 判为 acceptable，阈值在 train 上搜最优
  B4 CharNgramLR   - char 2-5 gram + TF-IDF + LogisticRegression
  B5 WordNgramLR   - word 1-2 gram + TF-IDF + LogisticRegression

评测指标与 LLM 评测完全一致：MCC（主）+ accuracy + 混淆矩阵。
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.pipeline import Pipeline


DATA_DIR = Path("cola_data")
OUT = Path("my_try/baseline_classical.json")
RNG_SEED = 42

COLS = ["source", "label", "first_label", "text"]


def load(split: str) -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / f"{split}.tsv",
        sep="\t",
        header=None,
        names=COLS,
        dtype={"source": str, "label": int, "first_label": str, "text": str},
        keep_default_na=False,
    )


def score(y_true: np.ndarray, y_pred: np.ndarray, name: str, elapsed: float) -> dict:
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    acc = (tp + tn) / len(y_true)
    pred_acc_rate = (tp + fp) / len(y_true)
    print(
        f"  {name:<18} MCC={mcc:+.4f}  Acc={acc:.4f}  "
        f"pred_acc_rate={pred_acc_rate:.3f}  elapsed={elapsed:.2f}s"
    )
    return {
        "name": name,
        "mcc": float(mcc),
        "accuracy": float(acc),
        "pred_acc_rate": float(pred_acc_rate),
        "confusion_matrix": {
            "labels": ["unacceptable(0)", "acceptable(1)"],
            "matrix": cm.tolist(),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "elapsed_sec": round(elapsed, 3),
    }


def b1_majority(train: pd.DataFrame, dev: pd.DataFrame) -> dict:
    t = time.time()
    majority = int(train["label"].mode()[0])
    y_pred = np.full(len(dev), majority, dtype=int)
    return score(dev["label"].values, y_pred, "B1_Majority", time.time() - t)


def b2_random(train: pd.DataFrame, dev: pd.DataFrame) -> dict:
    t = time.time()
    p = train["label"].mean()  # P(label=1)
    rng = np.random.default_rng(RNG_SEED)
    y_pred = (rng.random(len(dev)) < p).astype(int)
    return score(dev["label"].values, y_pred, "B2_Random", time.time() - t)


def b3_length_threshold(train: pd.DataFrame, dev: pd.DataFrame) -> dict:
    """假设：短句更易语法错。在 train 上搜最优 token 数阈值 k，
    规则：len(sent.split()) >= k → 预测 1(acceptable)，否则 0。"""
    t = time.time()
    train_lens = train["text"].str.split().str.len().values
    train_y = train["label"].values

    best_mcc = -2.0
    best_k = None
    for k in range(1, 30):
        y_hat = (train_lens >= k).astype(int)
        m = matthews_corrcoef(train_y, y_hat)
        if m > best_mcc:
            best_mcc = m
            best_k = k

    dev_lens = dev["text"].str.split().str.len().values
    y_pred = (dev_lens >= best_k).astype(int)
    r = score(dev["label"].values, y_pred, "B3_LengthThresh", time.time() - t)
    r["best_k"] = int(best_k)
    r["train_mcc"] = float(best_mcc)
    return r


def b4_char_ngram_lr(train: pd.DataFrame, dev: pd.DataFrame) -> dict:
    t = time.time()
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=200_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RNG_SEED,
                ),
            ),
        ]
    )
    pipe.fit(train["text"].values, train["label"].values)
    y_pred = pipe.predict(dev["text"].values)
    r = score(dev["label"].values, y_pred, "B4_CharNgramLR", time.time() - t)
    r["n_features"] = int(pipe.named_steps["tfidf"].vocabulary_.__len__())
    return r


def b5_word_ngram_lr(train: pd.DataFrame, dev: pd.DataFrame) -> dict:
    t = time.time()
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RNG_SEED,
                ),
            ),
        ]
    )
    pipe.fit(train["text"].values, train["label"].values)
    y_pred = pipe.predict(dev["text"].values)
    r = score(dev["label"].values, y_pred, "B5_WordNgramLR", time.time() - t)
    r["n_features"] = int(pipe.named_steps["tfidf"].vocabulary_.__len__())
    return r


def main() -> None:
    train = load("in_domain_train")
    dev = load("in_domain_dev")
    print(f"train: {len(train)} ({train['label'].mean():.3f} acc_rate)")
    print(f"dev  : {len(dev)} ({dev['label'].mean():.3f} acc_rate)")
    print()
    print("Running non-LLM baselines on CPU...")
    results = [
        b1_majority(train, dev),
        b2_random(train, dev),
        b3_length_threshold(train, dev),
        b4_char_ngram_lr(train, dev),
        b5_word_ngram_lr(train, dev),
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(
            {
                "dataset": "in_domain_dev.tsv",
                "n_train": len(train),
                "n_dev": len(dev),
                "train_acc_rate": float(train["label"].mean()),
                "dev_acc_rate": float(dev["label"].mean()),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n已保存：{OUT}")


if __name__ == "__main__":
    main()
