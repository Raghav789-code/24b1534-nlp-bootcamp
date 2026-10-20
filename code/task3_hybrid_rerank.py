"""Task 3 - hybrid (BM25 + dense) fusion and cross-encoder reranking.

Fusion weights are grid-searched on the TRAIN claims (all inputs are pretrained models, so train is
clean held-out data for them); the fusion method and reranker depth k are then chosen on val nDCG@10.
Outputs: predictions/t3_hybrid.trec, t3_rerank.trec (eval); cache/hybrid_*, rerank_* runs.
"""
import itertools
import json
import os
import time

import numpy as np

from common import (CACHE_DIR, PRED_DIR, evaluate, load_claims, load_corpus, load_run, log_metrics,
                    qrels_of, save_run, set_seed, topk_from_scores, update_results, write_trec)

SOURCES = ["bm25", "general", "scientific"]
RERANKERS = ["BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L-12-v2", "ncbi/MedCPT-Cross-Encoder"]
KS = [5, 10, 20, 30, 50, 100]


def load_scores(split):
    return {s: np.load(os.path.join(CACHE_DIR, f"{s}_{split}.npy")) for s in SOURCES}


def zscore(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def minmax(S):
    lo, hi = S.min(1, keepdims=True), S.max(1, keepdims=True)
    return (S - lo) / (hi - lo + 1e-9)


def rrf(S, k=60):
    ranks = np.argsort(np.argsort(-S, axis=1), axis=1) + 1
    return 1.0 / (k + ranks)


NORMS = {"zscore": zscore, "minmax": minmax, "rrf": rrf}


def fuse(scores, norm, w):
    return sum(wi * NORMS[norm](scores[s]) for s, wi in zip(SOURCES, w) if wi > 0)


def ids_qrels(split):
    cl = load_claims(split)
    return [int(c["id"]) for c in cl], qrels_of(cl), cl


def search_weights(norm, doc_ids, tr_scores, tr_ids, tr_q):
    # BM25 must take part (the task asks for a BM25 + dense hybrid), plus at least one dense model
    grid = [tuple(round(float(x), 1) for x in w) for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=3)
            if abs(sum(w) - 1) < 1e-6 and w[0] > 0 and (w[1] > 0 or w[2] > 0)]
    best = None
    for w in grid:
        m = evaluate(dict(zip(tr_ids, topk_from_scores(fuse(tr_scores, norm, w), doc_ids))), tr_q)
        if best is None or m["ndcg10"] > best[0]:
            best = (m["ndcg10"], w)
    return best


def main():
    set_seed()
    docs, doc_ids, _ = load_corpus()
    doc_text = {int(d["doc_id"]): d["title"] + ". " + " ".join(d["abstract"]) for d in docs}
    tr_ids, tr_q, _ = ids_qrels("train")
    va_ids, va_q, va_cl = ids_qrels("val")
    ev_ids, _, ev_cl = ids_qrels("eval")
    tr_s, va_s, ev_s = load_scores("train"), load_scores("val"), load_scores("eval")

    # ---------------- hybrid
    candidates = []
    for norm in NORMS:
        tr_ndcg, w = search_weights(norm, doc_ids, tr_s, tr_ids, tr_q)
        m = evaluate(dict(zip(va_ids, topk_from_scores(fuse(va_s, norm, w), doc_ids))), va_q)
        print(f"hybrid {norm} w(bm25,gen,sci)={w}: train nDCG={tr_ndcg:.4f}  val={m}")
        candidates.append((m["ndcg10"], norm, w, m))
    _, norm, w, m = max(candidates, key=lambda x: x[0])
    log_metrics("t3_hybrid", m)
    if norm == "rrf":
        method = (f"weighted reciprocal rank fusion over the full corpus: score = {w[0]}/(60+rank_BM25) + "
                  f"{w[1]}/(60+rank_BGE) + {w[2]}/(60+rank_MedCPT); weights grid-searched on train claims, "
                  f"RRF chosen over z-score / min-max score interpolation on val")
    else:
        method = (f"weighted sum of per-query {norm}-normalised scores over the full corpus: "
                  f"{w[0]}*BM25 + {w[1]}*BGE-base + {w[2]}*MedCPT (weights grid-searched on train claims, "
                  f"normalisation chosen on val)")
    t0 = time.time()
    for split, S, ids in [("val", va_s, va_ids), ("eval", ev_s, ev_ids), ("train", tr_s, tr_ids)]:
        ranked = topk_from_scores(fuse(S, norm, w), doc_ids)
        save_run(f"hybrid_{split}", ids, ranked)
        if split == "eval":
            write_trec(os.path.join(PRED_DIR, "t3_hybrid.trec"), ids, ranked, "hybrid")
    fusion_ms = (time.time() - t0) / (len(va_ids) + len(ev_ids) + len(tr_ids)) * 1000

    # ---------------- rerank the best first-stage system (by val nDCG@10)
    with open(os.path.join(os.path.dirname(__file__), "val_metrics.json")) as f:
        vm = json.load(f)
    first = max(["t1_bm25", "general", "scientific", "t3_hybrid"], key=lambda k: vm[k]["ndcg10"])
    first_run = {"t1_bm25": "bm25", "t3_hybrid": "hybrid"}.get(first, first)
    print("first stage for reranking:", first_run)

    best = None
    ce_scores = {}
    for name in RERANKERS:
        base = load_run(f"{first_run}_val")
        cache = os.path.join(CACHE_DIR, f"ce_{name.replace('/', '__')}_{first_run}_val.json")
        if os.path.exists(cache):
            with open(cache) as f:
                sc = {int(q): v for q, v in json.load(f).items()}
        else:
            ce = load_ce(name)
            claim = {int(c["id"]): c["claim"] for c in va_cl}
            sc = {q: ce.predict([(claim[q], doc_text[d]) for d, _ in base[q]], batch_size=32).tolist()
                  for q in va_ids}
            with open(cache, "w") as f:
                json.dump(sc, f)
            del ce
        ce_scores[name] = sc
        print(f"scored val top-100 with {name}", flush=True)
        for k in KS:
            m = evaluate(rerank(base, sc, k), va_q)
            print(f"rerank {name} k={k}: {m}")
            if best is None or m["ndcg10"] > best[0]["ndcg10"]:
                best = (m, name, k)
    m, name, k = best
    log_metrics("t3_rerank", m)

    # latency of the reranking stage alone (k pairs per claim, batch = k)
    ce = load_ce(name)
    base_val = load_run(f"{first_run}_val")
    claim = {int(c["id"]): c["claim"] for c in va_cl}
    ce.predict([(claim[va_ids[0]], doc_text[d]) for d, _ in base_val[va_ids[0]][:k]])  # warm-up
    t0 = time.time()
    for q in va_ids[:60]:
        ce.predict([(claim[q], doc_text[d]) for d, _ in base_val[q][:k]], batch_size=k)
    rerank_ms = (time.time() - t0) / 60 * 1000

    for split, cl in [("val", va_cl), ("eval", ev_cl)]:
        base = load_run(f"{first_run}_{split}")
        if split == "val" and name in ce_scores:
            sc = ce_scores[name]
        else:
            claim = {int(c["id"]): c["claim"] for c in cl}
            sc = {q: ce.predict([(claim[q], doc_text[d]) for d, _ in base[q][:k]], batch_size=32)
                  for q in base}
        rr = rerank(base, sc, k)
        ids = [int(c["id"]) for c in cl]
        save_run(f"rerank_{split}", ids, [rr[q] for q in ids])
        if split == "eval":
            write_trec(os.path.join(PRED_DIR, "t3_rerank.trec"), ids, [rr[q] for q in ids], "rerank")

    update_results("task3", {
        "hybrid_method": method,
        "reranker_model": f"{name} (cross-encoder over claim + title/abstract, max 512 tokens), "
                          f"first stage = {first_run}",
        "rerank_top_k": k,
        "val_ndcg10_hybrid": round(vm["t3_hybrid"]["ndcg10"] if "t3_hybrid" in vm else 0, 4),
        "val_ndcg10_rerank": round(m["ndcg10"], 4),
        "rerank_query_latency_ms": round(rerank_ms, 1),
    })
    with open(os.path.join(CACHE_DIR, "task3_meta.json"), "w") as f:
        json.dump({"first_stage": first_run, "fusion_ms_per_query": fusion_ms, "reranker": name, "k": k,
                   "rerank_ms": rerank_ms}, f)


def load_ce(name):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(name, max_length=512)
    if ce.model.device.type == "cuda":
        ce.model.half()
    return ce


def rerank(base, sc, k):
    """Top-k of base in cross-encoder order with scores shifted above the tail; tail keeps base order."""
    out = {}
    for q, docs in base.items():
        s = np.asarray(sc[q][:k], dtype=float)
        order = np.argsort(-s, kind="stable")
        tail = docs[k:]
        # map so that every head score > every tail score, keeping monotonic non-increasing scores
        tail_top = tail[0][1] if tail else 0.0
        head = [(docs[i][0], float(s[i] - s.min() + 1 + tail_top)) for i in order]
        out[q] = head + list(tail)
    return out


if __name__ == "__main__":
    main()
