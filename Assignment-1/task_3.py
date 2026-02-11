# Run pip install sklearn-crfsuite
# task_3.py
import re
from collections import Counter

import numpy as np

from utils import flatten_ner_sequences, token_accuracy, macro_f1_from_int_labels
from data_utils import load_conll2003


# -----------------------------
# Feature engineering
# -----------------------------

_RE_HAS_DIGIT = re.compile(r".*\d.*")
_RE_HAS_HYPHEN = re.compile(r".*-.*")

def word_shape(w: str) -> str:
    """
    Simple word shape: Xxxx, xxxx, ddd, etc.
    """
    out = []
    for ch in w:
        if ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        elif ch.isdigit():
            out.append("d")
        else:
            out.append(ch)
    # compress repeats
    shape = []
    for c in out:
        if not shape or shape[-1] != c:
            shape.append(c)
    return "".join(shape)

def token_features(sent_tokens, i: int):
    w = sent_tokens[i]
    w_lower = w.lower()

    feats = {
        "bias": 1.0,

        # lexical
        "w": w,
        "w.lower": w_lower,

        # shape
        "isupper": w.isupper(),
        "istitle": w.istitle(),
        "islower": w.islower(),
        "isdigit": w.isdigit(),
        "has_digit": bool(_RE_HAS_DIGIT.match(w)),
        "has_hyphen": bool(_RE_HAS_HYPHEN.match(w)),
        "shape": word_shape(w),

        # subword prefixes/suffixes
        "pref1": w_lower[:1],
        "pref2": w_lower[:2],
        "pref3": w_lower[:3],
        "pref4": w_lower[:4] if len(w_lower) >= 4 else w_lower,
        "suf1": w_lower[-1:],
        "suf2": w_lower[-2:] if len(w_lower) >= 2 else w_lower,
        "suf3": w_lower[-3:] if len(w_lower) >= 3 else w_lower,
        "suf4": w_lower[-4:] if len(w_lower) >= 4 else w_lower,
    }

    # context window: prev/next token lexical + a few shapes
    if i == 0:
        feats["BOS"] = True
    else:
        w_prev = sent_tokens[i - 1]
        feats.update({
            "-1:w.lower": w_prev.lower(),
            "-1:istitle": w_prev.istitle(),
            "-1:isupper": w_prev.isupper(),
            "-1:shape": word_shape(w_prev),
        })

    if i == len(sent_tokens) - 1:
        feats["EOS"] = True
    else:
        w_next = sent_tokens[i + 1]
        feats.update({
            "+1:w.lower": w_next.lower(),
            "+1:istitle": w_next.istitle(),
            "+1:isupper": w_next.isupper(),
            "+1:shape": word_shape(w_next),
        })

    return feats

def sent2features(tokens):
    return [token_features(tokens, i) for i in range(len(tokens))]


# -----------------------------
# Data conversion
# -----------------------------

def dataset_to_xy(split, label_names):
    """
    Converts HF dataset split to (X, y) where:
      X = list of sentences, each is list[dict] features
      y = list of sentences, each is list[int] labels
    """
    X = []
    y = []
    for ex in split:
        tokens = ex["tokens"]
        ner_tags = ex["ner_tags"]  # ints
        X.append(sent2features(tokens))
        y.append(list(ner_tags))
    return X, y


# -----------------------------
# Train / Eval CRF
# -----------------------------

def train_crf(X_train, y_train, X_dev, y_dev):
    import sklearn_crfsuite
    from sklearn_crfsuite import metrics

    # Light tuning candidates (keep it small)
    configs = [
        {"c1": 0.1, "c2": 0.1},
        {"c1": 0.05, "c2": 0.1},
        {"c1": 0.1, "c2": 0.05},
        {"c1": 0.0, "c2": 0.1},
        {"c1": 0.1, "c2": 0.0},
    ]

    best = None  # (macro_f1, cfg, model)
    for cfg in configs:
        crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=cfg["c1"],
            c2=cfg["c2"],
            max_iterations=200,
            all_possible_transitions=True,
        )
        crf.fit(X_train, y_train)

        y_pred = crf.predict(X_dev)
        yt, yp = flatten_ner_sequences(y_dev, y_pred)

        f1 = macro_f1_from_int_labels(yt, yp, num_classes=None)
        acc = token_accuracy(yt, yp)
        print(f"[dev] c1={cfg['c1']} c2={cfg['c2']}  acc={acc:.4f}  macroF1={f1:.4f}")

        if best is None or f1 > best[0]:
            best = (f1, cfg, crf)

    assert best is not None
    print("BEST CONFIG:", best[1], "dev_macroF1:", best[0])
    return best[2], best[1]


def most_important_features(crf, top_n: int = 30):
    """
    Returns top features by absolute weight for state features.
    """
    # sklearn-crfsuite exposes state_features_ / transition_features_
    state_items = list(crf.state_features_.items())  # ((attr, label), weight)
    state_items.sort(key=lambda x: abs(x[1]), reverse=True)

    top = state_items[:top_n]
    # format
    out = []
    for (attr, label), w in top:
        out.append((attr, label, float(w)))
    return out


def main():
    ds = load_conll2003()

    label_names = ds["train"].features["ner_tags"].feature.names
    num_labels = len(label_names)
    print("NER labels:", label_names)

    X_train, y_train = dataset_to_xy(ds["train"], label_names)
    X_dev, y_dev = dataset_to_xy(ds["validation"], label_names)
    X_test, y_test = dataset_to_xy(ds["test"], label_names)

    crf, cfg = train_crf(X_train, y_train, X_dev, y_dev)

    # Evaluate on test
    y_pred = crf.predict(X_test)
    yt, yp = flatten_ner_sequences(y_test, y_pred)

    acc = token_accuracy(yt, yp)
    f1 = macro_f1_from_int_labels(yt, yp, num_classes=num_labels)

    print(f"\n[TEST] acc={acc:.4f}  macroF1={f1:.4f}")

    # Feature importance
    top_feats = most_important_features(crf, top_n=40)
    print("\nTop state features (by |weight|):")
    for attr, label, w in top_feats:
        print(f"{w:+.4f}\t{label}\t{attr}")

    # You can also inspect transition features if you want:
    # trans = list(crf.transition_features_.items())
    # trans.sort(key=lambda x: abs(x[1]), reverse=True)
    # print("\nTop transition features:")
    # for (l1, l2), w in trans[:30]:
    #     print(f"{w:+.4f}\t{l1} -> {l2}")


if __name__ == "__main__":
    main()
