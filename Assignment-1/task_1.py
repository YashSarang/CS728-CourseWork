# task_1.py
import os
import time
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from utils import Vocab, build_cooccurrence_matrix, top_k_cosine_neighbors, set_global_seed
from data_utils import load_ccnews_json, extract_vocab_from_ccnews_json, load_ccnews_documents_as_tokens


def glove_weight(x: np.ndarray, x_max: float, alpha: float) -> np.ndarray:
    w = (x / x_max) ** alpha
    w = np.minimum(w, 1.0)
    return w


def train_glove_sgd(
    X: sparse.csr_matrix,
    d: int,
    lr: float,
    iters: int,
    x_max: float = 100.0,
    alpha: float = 0.75,
    seed: int = 1337,
    report_every: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Train GloVe using SGD over nonzero co-occurrence entries.
    Returns W, W_tilde, b, b_tilde, losses(list per iter).
    """
    set_global_seed(seed)

    Xcoo = X.tocoo()
    I = Xcoo.row.astype(np.int64)
    J = Xcoo.col.astype(np.int64)
    Xij = Xcoo.data.astype(np.float32)

    # Filter out zeros (shouldn't exist) and very tiny weights if needed (optional)
    # Keep as-is for assignment consistency.

    logX = np.log(Xij + 1e-12)
    wts = glove_weight(Xij, x_max=x_max, alpha=alpha).astype(np.float32)

    V = X.shape[0]
    W = 0.01 * np.random.randn(V, d).astype(np.float32)
    Wt = 0.01 * np.random.randn(V, d).astype(np.float32)
    b = np.zeros((V,), dtype=np.float32)
    bt = np.zeros((V,), dtype=np.float32)

    losses = []

    n = len(Xij)
    idx = np.arange(n)

    for it in range(1, iters + 1):
        t0 = time.time()

        # Shuffle samples each epoch
        np.random.shuffle(idx)

        total_loss = 0.0

        for k in idx:
            i = I[k]
            j = J[k]
            x = Xij[k]
            w = wts[k]
            target = logX[k]

            # prediction
            dot = float(np.dot(W[i], Wt[j]))
            pred = dot + float(b[i]) + float(bt[j])
            err = pred - float(target)

            # weighted squared error
            loss = w * (err * err)
            total_loss += float(loss)

            # gradients
            # d/dW[i] = 2*w*err*Wt[j]
            # d/dWt[j] = 2*w*err*W[i]
            g = 2.0 * w * err

            Wi = W[i].copy()
            W[i] -= lr * (g * Wt[j])
            Wt[j] -= lr * (g * Wi)
            b[i] -= lr * g
            bt[j] -= lr * g

        avg_loss = total_loss / max(n, 1)
        losses.append(avg_loss)

        dt = time.time() - t0
        if report_every and (it % report_every == 0):
            print(f"[iter {it:03d}/{iters}] avg_loss={avg_loss:.6f}  time={dt:.2f}s  nnz={n}")

    return W, Wt, b, bt, losses


def save_loss_plot(losses: list[float], outpath: str, title: str):
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Avg Weighted Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_embeddings(path: str, vocab: Vocab, E: np.ndarray):
    """
    Save as .npz (fast) + optional text.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=E, id_to_token=np.array(vocab.id_to_token, dtype=object))


def main():
    # ---- Edit these paths ----
    CCNEWS_JSON = "data/cc_news_filtered.json"  # put your actual path
    ASSETS_DIR = "assets"
    OUT_DIR = "outputs"

    # ---- Hyperparam search space (d fixed at 200) ----
    d_fixed = 200
    windows = [2, 5, 10]
    lrs = [0.05, 0.025, 0.01]
    iters = 30
    x_max = 100.0
    alpha = 0.75

    # ---- Load dataset + vocab ----
    payload = load_ccnews_json(CCNEWS_JSON)
    vocab_tokens = extract_vocab_from_ccnews_json(payload)
    vocab = Vocab.from_tokens(vocab_tokens, add_unk=True)  # add_unk only helps Task4 later

    docs_tok = load_ccnews_documents_as_tokens(CCNEWS_JSON, vocab=vocab, lowercase=True)

    # ---- Track results ----
    results_csv = Path(ASSETS_DIR) / "glove_search_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["w", "lr", "iters", "x_max", "alpha", "final_loss", "total_train_s", "nnz"])

        best = None  # (final_loss, config_dict, model_params)
        for w in windows:
            # build co-occurrence once per window
            t_build0 = time.time()
            X = build_cooccurrence_matrix(
                docs_tok,
                vocab=vocab,
                window_size=w,
                symmetric=True,
                distance_weighting=False,  # conservative for GloVe
            )
            build_s = time.time() - t_build0
            print(f"[build] w={w} X.nnz={X.nnz} time={build_s:.2f}s")

            for lr in lrs:
                t0 = time.time()
                W, Wt, b, bt, losses = train_glove_sgd(
                    X=X,
                    d=d_fixed,
                    lr=lr,
                    iters=iters,
                    x_max=x_max,
                    alpha=alpha,
                    seed=1337,
                )
                train_s = time.time() - t0

                final_loss = losses[-1]
                writer.writerow([w, lr, iters, x_max, alpha, final_loss, train_s, X.nnz])
                f.flush()

                plot_path = str(Path(ASSETS_DIR) / f"glove_loss_w{w}_d{d_fixed}_lr{lr}.png")
                save_loss_plot(
                    losses,
                    outpath=plot_path,
                    title=f"GloVe Loss (w={w}, d={d_fixed}, lr={lr})",
                )

                if (best is None) or (final_loss < best[0]):
                    best = (final_loss, {"w": w, "lr": lr, "iters": iters, "x_max": x_max, "alpha": alpha}, (W, Wt, b, bt, X))

                print(f"[done] w={w} lr={lr} final_loss={final_loss:.6f} train_time={train_s:.2f}s")

    # ---- Best config ----
    assert best is not None
    best_loss, cfg, (W, Wt, b, bt, Xbest) = best
    print("BEST:", cfg, "loss:", best_loss)

    # ---- Final embedding for nearest neighbors ----
    E200 = W + Wt  # common choice
    chosen_words = ["united", "city", "president"]  # replace with your 3 words (must be in vocab)
    for qw in chosen_words:
        nn = top_k_cosine_neighbors(qw, vocab=vocab, embeddings=E200, k=5, exclude_self=True)
        print(qw, "->", nn)

    # Save best 200-d embeddings
    save_embeddings(str(Path(OUT_DIR) / "glove_best_d200.npz"), vocab, E200)

    # ---- After you pick cfg, re-run for required dimensions ----
    # Example: if required d are [50, 100, 200, 300] (adjust to actual required list)
    required_ds = [50, 100, 200, 300]
    for d in required_ds:
        if d == d_fixed:
            continue
        print(f"\n[final train] d={d} using cfg={cfg}")
        # rebuild X for cfg["w"] (or reuse Xbest if same w)
        X = Xbest if cfg["w"] == cfg["w"] else Xbest
        W, Wt, b, bt, losses = train_glove_sgd(
            X=X,
            d=d,
            lr=cfg["lr"],
            iters=cfg["iters"],
            x_max=cfg["x_max"],
            alpha=cfg["alpha"],
            seed=1337,
        )
        E = W + Wt
        save_embeddings(str(Path(OUT_DIR) / f"glove_d{d}.npz"), vocab, E)


if __name__ == "__main__":
    main()
