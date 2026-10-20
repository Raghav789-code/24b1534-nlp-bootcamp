"""Q1 per-claim comparison (val_bm25.trec vs val_general.trec) + validation ablation table."""
import json
import os

from common import PRED_DIR, ROOT, load_claims, load_corpus, per_claim_ndcg, qrels_of, read_trec, update_results


def main():
    val = load_claims("val")
    qrels = qrels_of(val)
    claim = {int(c["id"]): c["claim"] for c in val}
    docs, _, _ = load_corpus()
    title = {int(d["doc_id"]): d["title"] for d in docs}
    bm = read_trec(os.path.join(PRED_DIR, "val_bm25.trec"))
    ge = read_trec(os.path.join(PRED_DIR, "val_general.trec"))
    nb, ng = per_claim_ndcg(bm, qrels), per_claim_ndcg(ge, qrels)
    diff = sorted(qrels, key=lambda q: nb[q] - ng[q])
    bm_wins, ge_wins = diff[::-1][:2], diff[:2]
    print(f"BM25 better on {sum(nb[q] > ng[q] for q in qrels)} claims, BGE better on "
          f"{sum(ng[q] > nb[q] for q in qrels)}, ties {sum(nb[q] == ng[q] for q in qrels)}")
    for tag, qs in [("BM25 wins", bm_wins), ("general wins", ge_wins)]:
        for q in qs:
            gold = list(qrels[q])
            rb = [d for d, _ in bm[q]].index(gold[0]) + 1 if gold[0] in [d for d, _ in bm[q]] else ">100"
            rg = [d for d, _ in ge[q]].index(gold[0]) + 1 if gold[0] in [d for d, _ in ge[q]] else ">100"
            print(f"\n[{tag}] claim {q}: nDCG bm25={nb[q]:.3f} general={ng[q]:.3f} "
                  f"(gold rank bm25={rb}, general={rg})\n  claim: {claim[q]}\n  gold:  {title[gold[0]]}"
                  f"\n  bm25 top1:    {title[bm[q][0][0]]}\n  general top1: {title[ge[q][0][0]]}")
    update_results("report_q1", {"bm25_wins": bm_wins, "general_wins": ge_wins})

    with open(os.path.join(ROOT, "code", "val_metrics.json")) as f:
        vm = json.load(f)
    print("\nValidation ablation")
    print(f"{'run':14s} {'nDCG@10':>8s} {'R@100':>8s} {'MRR':>8s}")
    for k, m in vm.items():
        print(f"{k:14s} {m['ndcg10']:8.4f} {m['recall100']:8.4f} {m['mrr']:8.4f}")


if __name__ == "__main__":
    main()
