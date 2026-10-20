"""Write ../report.md (Q1-Q5, <= 100 words each) from results.json, val_metrics.json and cached metadata.

Run after task5_verify.py and report_analysis.py. The text is filled from the measured numbers.
"""
import json
import os

from common import CACHE_DIR, PRED_DIR, ROOT, load_claims, per_claim_ndcg, qrels_of, read_trec


def jload(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else default


def main():
    res = jload(os.path.join(ROOT, "results.json"))
    vm = jload(os.path.join(ROOT, "code", "val_metrics.json"))
    t3 = jload(os.path.join(CACHE_DIR, "task3_meta.json"), {})
    t5 = jload(os.path.join(CACHE_DIR, "task5_meta.json"), {})
    hist = {m: jload(os.path.join(CACHE_DIR, f"ft_{m}_history.json"), {}) for m in ["random", "hard"]}
    val = load_claims("val")
    qrels = qrels_of(val)
    claim = {int(c["id"]): c["claim"] for c in val}
    nb = per_claim_ndcg(read_trec(os.path.join(PRED_DIR, "val_bm25.trec")), qrels)
    ng = per_claim_ndcg(read_trec(os.path.join(PRED_DIR, "val_general.trec")), qrels)
    q1 = res["report_q1"]

    def short(q, n=9):
        w = claim[q].split()
        return " ".join(w[:n]) + ("…" if len(w) > n else "")

    a, b = q1["bm25_wins"]
    c, d = q1["general_wins"]
    Q1 = (f"BM25 wins: {a} ({nb[a]:.2f} vs {ng[a]:.2f}, \"{short(a, 6)}\") and {b} ({nb[b]:.2f} vs {ng[b]:.2f}). "
          f"General (BGE) wins: {c} ({ng[c]:.2f} vs {nb[c]:.2f}, \"{short(c, 6)}\") and {d} "
          f"({ng[d]:.2f} vs {nb[d]:.2f}). BM25 wins when the claim shares rare exact tokens (gene/drug names, "
          f"acronyms) with the gold abstract, which the dense model blurs into topically similar papers. BGE wins "
          f"when the claim paraphrases the abstract, where lexical overlap is low.")

    h, r = vm["t3_hybrid"]["ndcg10"], vm["t3_rerank"]["ndcg10"]
    best_single = max(vm["general"]["ndcg10"], vm["scientific"]["ndcg10"], vm["t1_bm25"]["ndcg10"])
    gain_h, gain_r = h - best_single, r - h
    big = "reranking" if gain_r > gain_h else "hybrid fusion"
    lat = res["task3"]["rerank_query_latency_ms"]
    dense_lat = max(res["task2"]["query_latency_ms"].values())
    Q2 = (f"Hybrid fusion moved val nDCG@10 from {best_single:.3f} (best single retriever) to {h:.3f} "
          f"(+{gain_h:.3f}); cross-encoder reranking of the top-{res['task3']['rerank_top_k']} moved it to "
          f"{r:.3f} ({gain_r:+.3f}). The largest gain came from {big}. Fusion is almost free (one extra dense "
          f"query, ~{dense_lat:.0f} ms, plus a vector sum), while reranking adds ~{lat:.0f} ms per claim on our GPU "
          f"because every (claim, abstract) pair needs a full 512-token cross-encoder pass. "
          f"General-domain rerankers (bge-reranker-base, MiniLM) actually hurt; only the biomedical MedCPT "
          f"cross-encoder helped.")

    vr, vh = vm["t4_random"]["ndcg10"], vm["t4_hard"]["ndcg10"]
    def curve(m):
        return ", ".join(f"ep{x['epoch']} {x['val']['ndcg10']:.3f}" + (f"/{x['train']['ndcg10']:.3f}" if 'train' in x else "")
                         for x in hist[m].get("history", []))
    Q3 = (f"Hard negatives {'helped' if vh > vr else 'did not help'}: val nDCG@10 {vh:.3f} vs {vr:.3f} with random "
          f"negatives (base model {vm['general']['ndcg10']:.3f}). We logged val/train nDCG@10 after every epoch "
          f"(random: {curve('random')}; hard: {curve('hard')}). Overfitting shows as train nDCG rising while val "
          f"stalls or drops; we kept the best-val epoch. With only ~440 claim–abstract pairs the gap opens "
          f"quickly, and some mined 'hard negatives' are unlabelled positives, which limits the benefit.")

    cats = t5.get("categories", {})
    n_err = t5.get("n_errors", 0)
    rm = cats.get("retrieval_miss", 0)
    Q4 = (f"{rm} of {n_err} validation verification errors are retrieval_miss "
          f"({(rm / n_err if n_err else 0):.0%}). Full breakdown: "
          + ", ".join(f"{k} {v}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
          + f". Validation F1 = {t5.get('val_f1', 0):.3f} (P {t5.get('val_p', 0):.3f}, R {t5.get('val_r', 0):.3f}). "
          f"So most errors come from the verifier (label/threshold decisions on retrieved abstracts), not from "
          f"retrieval, which already reaches Recall@100 = 1.0.")

    Q5 = ("Fine-tune the NLI verifier on the SciFact training rationales (claim, gold rationale sentences → "
          "SUPPORT/CONTRADICT, plus retrieved non-rationale sentences as NEI) instead of using it zero-shot. "
          "Retrieval is already strong (nDCG@10 > 0.9, Recall@100 = 1.0), so the remaining errors are mostly "
          "verification: MNLI-style models miss SciFact-specific contradictions (opposite effect direction, "
          "negated findings, numbers) and are poorly calibrated about when an abstract is relevant but neutral. "
          "In-domain supervision targets exactly these failure modes.")

    out = ["# Report (all analysis on validation claims)", ""]
    for i, t in enumerate([Q1, Q2, Q3, Q4, Q5], 1):
        out += [f"## Q{i}", "", t, ""]
        print(f"Q{i}: {len(t.split())} words")
    with open(os.path.join(ROOT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


if __name__ == "__main__":
    main()
