"""Task 5 - claim verification over the best retriever's top abstracts + error analysis.

Pipeline per (claim, retrieved abstract):
  1. zero-shot NLI (DeBERTa-v3-base MNLI/FEVER/ANLI) on every abstract sentence (premise) vs the claim
     (hypothesis), plus once on the whole abstract;
  2. features = max/top-2 entailment & contradiction over sentences, abstract-level probs, retrieval rank;
  3. a 3-way logistic regression (SUPPORT / CONTRADICT / NONE) trained on the TRAIN claims, whose
     decision margin and number of abstracts considered are tuned with 5-fold out-of-fold predictions
     on train. Validation is only used to report the final number and for the error analysis.
Outputs: predictions/t5_verification.csv, t5_val_verification.csv, t5_val_errors.csv
"""
import csv
import json
import os
import re

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from common import CACHE_DIR, PRED_DIR, ROOT, load_claims, load_corpus, load_run, set_seed, update_results
from encoders import DEVICE

NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
CAND = 3          # abstracts per claim passed to the verifier (max rows allowed per claim)
LABELS = ["NONE", "SUPPORT", "CONTRADICT"]
NEG_RE = re.compile(r"\b(not|no|never|none|without|fail|fails|failed|cannot|lack|lacks|absence|"
                    r"neither|nor|unable|independent|unaffected|n't)\b", re.I)
NUM_RE = re.compile(r"\d|percent|%|fold|times|half|double|twice|one|two|three|ten|hundred|thousand", re.I)


class NLIScorer:
    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(NLI)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI).to(DEVICE).eval()
        l2i = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.idx = [l2i["entailment"], l2i["neutral"], l2i["contradiction"]]
        cache = os.path.join(CACHE_DIR, "nli_cache.json")
        self.cache_path = cache
        self.cache = json.load(open(cache)) if os.path.exists(cache) else {}

    @torch.no_grad()
    def probs(self, premises, hyp, bs=32):
        out = []
        for i in range(0, len(premises), bs):
            x = self.tok(premises[i:i + bs], [hyp] * len(premises[i:i + bs]), truncation="only_first",
                         max_length=512, padding=True, return_tensors="pt").to(DEVICE)
            p = torch.softmax(self.model(**x).logits.float(), -1)[:, self.idx]
            out.append(p.cpu().numpy())
        return np.concatenate(out)          # columns: entail, neutral, contra

    def doc(self, claim, d):
        key = f"{claim}|||{d['doc_id']}"
        if key not in self.cache:
            sent = self.probs(d["abstract"], claim)
            whole = self.probs([d["title"] + ". " + " ".join(d["abstract"])], claim)[0]
            self.cache[key] = {"sent": sent.tolist(), "whole": whole.tolist()}
        return np.asarray(self.cache[key]["sent"]), np.asarray(self.cache[key]["whole"])

    def save(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)


def features(sent, whole, rank):
    e, c = sent[:, 0], sent[:, 2]
    es, cs = np.sort(e)[::-1], np.sort(c)[::-1]
    best = sent[np.argmax(e + c)]
    return [es[0], es[:2].mean(), cs[0], cs[:2].mean(), best[0], best[2], es[0] - cs[0],
            whole[0], whole[1], whole[2], 1.0 / rank, float(rank == 1)]


def build(split, run, nli, by_id, train=False):
    """Rows of (claim_id, doc_id, rank, features, label_idx)."""
    rows = []
    for c in load_claims(split):
        q = int(c["id"])
        gold = {int(e["doc_id"]): e["label"] for e in c.get("evidence") or []}
        docs = [d for d, _ in run[q][:CAND]]
        if train:          # also learn from gold abstracts the retriever missed
            docs += [g for g in gold if g not in docs]
        for r, d in enumerate(docs, 1):
            sent, whole = nli.doc(c["claim"], by_id[d])
            y = LABELS.index(gold[d]) if d in gold else 0
            rows.append((q, d, min(r, CAND + 1), features(sent, whole, min(r, CAND + 1)), y))
    return rows


def decide(rows, proba, margin, n_docs):
    """Pick label per (claim, doc); emit SUPPORT/CONTRADICT only if it beats NONE by `margin`."""
    preds = []
    for (q, d, r, _, _), p in zip(rows, proba):
        if r > n_docs:
            continue
        lab = 1 + int(np.argmax(p[1:]))
        if p[lab] - p[0] > margin:
            preds.append((q, d, LABELS[lab], p[lab]))
    out = {}
    for q, d, lab, s in sorted(preds, key=lambda x: -x[3]):
        if len(out.setdefault(q, [])) < 3:
            out[q].append((d, lab))
    return {(q, d, lab) for q, v in out.items() for d, lab in v}


def gold_triples(claims):
    return {(int(c["id"]), int(e["doc_id"]), e["label"]) for c in claims for e in c.get("evidence") or []}


def prf(pred, gold):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def categorize(c, pred_pairs, retrieved, by_id):
    gold = {int(e["doc_id"]): e["label"] for e in c["evidence"]}
    claim = c["claim"]
    if gold and any(g not in retrieved for g in gold):
        return "retrieval_miss"
    correct = {d for d, l in pred_pairs.items() if gold.get(d) == l}
    if len(gold) > 1 and correct and len(correct) < len(gold):
        return "needs_multiple_docs"
    flipped = [d for d, l in pred_pairs.items() if d in gold and gold[d] != l]
    missed = [d for d in gold if d not in pred_pairs]
    false_pos = [d for d in pred_pairs if d not in gold]
    if flipped or missed:
        d = (flipped or missed)[0]
        rat = " ".join(by_id[d]["abstract"][i] for e in c["evidence"] if int(e["doc_id"]) == d
                       for i in e.get("sentences", []) if i < len(by_id[d]["abstract"]))
        if NEG_RE.search(claim) or (flipped and NEG_RE.search(rat)):
            return "negation"
        if NUM_RE.search(claim):
            return "numerical"
        return "other"
    if false_pos:
        return "numerical" if re.search(r"\d", claim) else "entity_mismatch"
    return "other"


def main():
    set_seed()
    docs, _, _ = load_corpus()
    by_id = {int(d["doc_id"]): d for d in docs}
    with open(os.path.join(ROOT, "code", "val_metrics.json")) as f:
        vm = json.load(f)
    names = {"t1_bm25": "bm25", "general": "general", "scientific": "scientific", "t3_hybrid": "hybrid",
             "t3_rerank": "rerank", "t4_random": "ft_random", "t4_hard": "ft_hard"}
    best_key = max((k for k in names if k in vm), key=lambda k: vm[k]["ndcg10"])
    retr = names[best_key]
    print(f"best retriever by val nDCG@10: {best_key} ({vm[best_key]['ndcg10']:.4f})")

    nli = NLIScorer()
    # train candidates come from the (pretrained-only) hybrid run: fine-tuned retrievers saw these claims
    train_rows = build("train", load_run("hybrid_train"), nli, by_id, train=True)
    val_rows = build("val", load_run(f"{retr}_val"), nli, by_id)
    eval_rows = build("eval", load_run(f"{retr}_eval"), nli, by_id)
    nli.save()

    X = np.array([r[3] for r in train_rows]); y = np.array([r[4] for r in train_rows])
    groups = np.array([r[0] for r in train_rows])
    clf_args = dict(C=1.0, max_iter=2000, class_weight="balanced")
    oof = np.zeros((len(y), 3))
    for tr, te in GroupKFold(5).split(X, y, groups):
        oof[te] = LogisticRegression(**clf_args).fit(X[tr], y[tr]).predict_proba(X[te])
    train_cl = load_claims("train")
    g_train = gold_triples(train_cl)
    best = None
    for margin in np.arange(-0.3, 0.61, 0.05):
        for n_docs in [1, 2, 3]:
            # only evaluate on rows a retriever would actually produce (rank <= CAND)
            f1 = prf(decide(train_rows, oof, margin, n_docs), g_train)[2]
            if best is None or f1 > best[0]:
                best = (f1, round(float(margin), 2), n_docs)
    print(f"train OOF F1={best[0]:.4f} margin={best[1]} n_docs={best[2]}")
    _, margin, n_docs = best
    clf = LogisticRegression(**clf_args).fit(X, y)

    val_cl = load_claims("val")
    vp = decide(val_rows, clf.predict_proba(np.array([r[3] for r in val_rows])), margin, n_docs)
    p, r, f1 = prf(vp, gold_triples(val_cl))
    print(f"val verification P={p:.4f} R={r:.4f} F1={f1:.4f}")
    ep = decide(eval_rows, clf.predict_proba(np.array([r[3] for r in eval_rows])), margin, n_docs)

    for name, triples in [("t5_verification.csv", ep), ("t5_val_verification.csv", vp)]:
        with open(os.path.join(PRED_DIR, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["claim_id", "doc_id", "label"])
            w.writerows(sorted(triples))

    # ---------------- error analysis on validation claims
    val_run = load_run(f"{retr}_val")
    errors = []
    for c in val_cl:
        q = int(c["id"])
        pred_pairs = {d: l for qq, d, l in vp if qq == q}
        gold_pairs = {int(e["doc_id"]): e["label"] for e in c["evidence"]}
        if pred_pairs != gold_pairs:
            retrieved = [d for d, _ in val_run[q][:n_docs]]
            errors.append((q, categorize(c, pred_pairs, retrieved, by_id)))
    with open(os.path.join(PRED_DIR, "t5_val_errors.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "category"])
        w.writerows(errors)
    from collections import Counter
    cnt = Counter(c for _, c in errors)
    print(f"{len(errors)} / {len(val_cl)} val claims wrong; categories: {dict(cnt)}")
    with open(os.path.join(CACHE_DIR, "task5_meta.json"), "w") as f:
        json.dump({"retriever": best_key, "margin": margin, "n_docs": n_docs, "train_oof_f1": best[0],
                   "val_p": p, "val_r": r, "val_f1": f1, "n_errors": len(errors),
                   "categories": dict(cnt), "n_val_pred": len(vp), "n_eval_pred": len(ep)}, f, indent=2)

    update_results("task5", {
        "retriever_used": f"{best_key} (best val nDCG@10 = {vm[best_key]['ndcg10']:.4f}); top-{n_docs} abstracts",
        "nli_model": f"{NLI} (184M), sentence-level + whole-abstract NLI, aggregated by a 3-way logistic "
                     f"regression trained on train claims",
        "zero_shot_or_finetuned": "zero-shot NLI model; decision layer (logistic regression + margin) "
                                  "trained/tuned on the train claims only",
        "val_f1": round(f1, 4),
    })


if __name__ == "__main__":
    main()
