"""Shared helpers: data loading, TREC I/O, metrics, timing."""
import json
import math
import os
import random

import numpy as np

SEED = 42
TOP_K = 100

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # submission/
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(ROOT), "data"))
PRED_DIR = os.path.join(ROOT, "predictions")
CACHE_DIR = os.path.join(ROOT, "cache")           # not shipped in the zip
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_corpus():
    docs = read_jsonl(os.path.join(DATA_DIR, "corpus.jsonl"))
    doc_ids = [int(d["doc_id"]) for d in docs]
    texts = [d["title"] + " " + " ".join(d["abstract"]) for d in docs]
    return docs, doc_ids, texts


def load_claims(split):
    return read_jsonl(os.path.join(DATA_DIR, f"{split}_claims.jsonl"))


def qrels_of(claims):
    """{claim_id: {doc_id: 1}} for claims with evidence (NEI claims are skipped, as in grading)."""
    q = {}
    for c in claims:
        ev = c.get("evidence") or []
        if ev:
            q[int(c["id"])] = {int(e["doc_id"]): 1 for e in ev}
    return q


# ---------------------------------------------------------------- runs
def topk_from_scores(score_matrix, doc_ids, k=TOP_K):
    """score_matrix: (n_queries, n_docs) -> list of [(doc_id, score), ...] sorted desc."""
    out = []
    for row in score_matrix:
        idx = np.argpartition(-row, k)[:k]
        idx = idx[np.argsort(-row[idx], kind="stable")]
        out.append([(doc_ids[i], float(row[i])) for i in idx])
    return out


def write_trec(path, claim_ids, ranked, run_name):
    with open(path, "w", encoding="utf-8") as f:
        for cid, docs in zip(claim_ids, ranked):
            assert len(docs) == TOP_K and len({d for d, _ in docs}) == TOP_K, cid
            prev = math.inf
            for r, (did, s) in enumerate(docs, 1):
                s = min(s, prev)          # scores must be non-increasing
                prev = s
                f.write(f"{cid} Q0 {did} {r} {s:.6f} {run_name}\n")


def read_trec(path):
    run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            run.setdefault(int(p[0]), []).append((int(p[3]), int(p[2]), float(p[4])))
    return {q: [(d, s) for _, d, s in sorted(v)] for q, v in run.items()}


def save_run(name, claim_ids, ranked):
    """Cache a full run (as dict) so later tasks can reuse it."""
    with open(os.path.join(CACHE_DIR, name + ".json"), "w") as f:
        json.dump({str(c): r for c, r in zip(claim_ids, ranked)}, f)


def load_run(name):
    with open(os.path.join(CACHE_DIR, name + ".json")) as f:
        return {int(c): [(int(d), float(s)) for d, s in r] for c, r in json.load(f).items()}


# ---------------------------------------------------------------- metrics
def ndcg_at_k(ranked_ids, rel, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked_ids, rel, k=100):
    return len(set(ranked_ids[:k]) & set(rel)) / len(rel)


def mrr(ranked_ids, rel, k=100):
    for i, d in enumerate(ranked_ids[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(run, qrels):
    """run: {claim_id: [(doc_id, score), ...]}. Averages over claims with gold evidence."""
    n, r, m = [], [], []
    for q, rel in qrels.items():
        ids = [d for d, _ in run[q]]
        n.append(ndcg_at_k(ids, rel))
        r.append(recall_at_k(ids, rel))
        m.append(mrr(ids, rel))
    return {"ndcg10": float(np.mean(n)), "recall100": float(np.mean(r)), "mrr": float(np.mean(m))}


def per_claim_ndcg(run, qrels):
    return {q: ndcg_at_k([d for d, _ in run[q]], rel) for q, rel in qrels.items()}


# ---------------------------------------------------------------- results.json
RESULTS_PATH = os.path.join(ROOT, "results.json")


def update_results(section, values):
    with open(RESULTS_PATH, encoding="utf-8") as f:
        res = json.load(f)
    if isinstance(res.get(section), dict):
        res[section].update(values)
    else:
        res[section] = values
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


def log_metrics(name, metrics):
    path = os.path.join(ROOT, "code", "val_metrics.json")
    allm = {}
    if os.path.exists(path):
        with open(path) as f:
            allm = json.load(f)
    allm[name] = metrics
    with open(path, "w") as f:
        json.dump(allm, f, indent=2)
    print(f"[val] {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
