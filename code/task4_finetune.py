"""Task 4 - fine-tune BGE-base on the training claims, once with random and once with hard negatives.

Loss: MultipleNegativesRankingLoss (in-batch negatives + one explicit negative per pair).
  random : explicit negative = random corpus abstract that is not gold for the claim
  hard   : explicit negative = sampled from the top-30 of BM25 / BGE / MedCPT on the claim (non-gold)
After every epoch we evaluate val nDCG@10 (and train nDCG@10) to detect overfitting and keep the best
epoch's checkpoint.
Outputs: predictions/t4_random.trec, t4_hard.trec (eval); cache/ft_{random,hard}_* runs.
"""
import json
import os
import random
import sys

import numpy as np
import torch

from common import (CACHE_DIR, PRED_DIR, SEED, evaluate, load_claims, load_corpus, load_run, log_metrics,
                    qrels_of, set_seed, topk_from_scores, update_results)
from encoders import BGE_PREFIX, DEVICE

BASE = "BAAI/bge-base-en-v1.5"
EPOCHS = 2
BATCH = 16
LR = 1e-5   # 2e-5 overfit after one epoch (val nDCG@10 0.891 -> 0.876 -> 0.865 while train rose)
MAX_LEN = 256


def doc_str(d):
    return d["title"] + ". " + " ".join(d["abstract"])


def build_examples(mode, docs, train):
    rng = random.Random(SEED)
    by_id = {int(d["doc_id"]): d for d in docs}
    all_ids = list(by_id)
    mined = {}
    if mode == "hard":
        runs = [load_run(f"{s}_train") for s in ["bm25", "general", "scientific"]]
        for c in train:
            q = int(c["id"])
            gold = {int(e["doc_id"]) for e in c["evidence"]}
            cands = []
            for r in runs:
                cands += [d for d, _ in r[q][:30] if d not in gold and d not in cands]
            mined[q] = cands
    ex = []
    for c in train:
        q = int(c["id"])
        gold = {int(e["doc_id"]) for e in c["evidence"]}
        for g in gold:
            if mode == "hard":
                neg = rng.choice(mined[q][:20])   # hard but not only the very top (false-negative risk)
            else:
                neg = rng.choice(all_ids)
                while neg in gold:
                    neg = rng.choice(all_ids)
            ex.append((BGE_PREFIX + c["claim"], doc_str(by_id[g]), doc_str(by_id[neg])))
    return ex


def evaluate_model(model, docs, doc_ids, splits=("val", "train")):
    """One corpus encoding (fp16 autocast, small batches to stay inside GPU memory), metrics per split."""
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        D = model.encode([doc_str(d) for d in docs], batch_size=16, normalize_embeddings=True,
                         convert_to_numpy=True)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    out = {}
    for split in splits:
        cl = load_claims(split)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            Q = model.encode([BGE_PREFIX + c["claim"] for c in cl], batch_size=64, normalize_embeddings=True,
                             convert_to_numpy=True)
        D, Q = D.astype("float32"), Q.astype("float32")
        ids = [int(c["id"]) for c in cl]
        out[split] = evaluate(dict(zip(ids, topk_from_scores(Q @ D.T, doc_ids))), qrels_of(cl))
    return out


def train(mode):
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    set_seed()
    docs, doc_ids, _ = load_corpus()
    train_cl = [c for c in load_claims("train") if c["evidence"]]
    ex = build_examples(mode, docs, train_cl)
    print(f"{mode}: {len(ex)} training triples from {len(train_cl)} claims")

    model = SentenceTransformer(BASE, device=DEVICE)
    model.max_seq_length = MAX_LEN
    loader = DataLoader([InputExample(texts=list(t)) for t in ex], shuffle=True, batch_size=BATCH,
                        generator=torch.Generator().manual_seed(SEED))
    loss = losses.MultipleNegativesRankingLoss(model)

    out_dir = os.path.join(CACHE_DIR, f"ft_{mode}")
    model.max_seq_length = 512
    e0 = evaluate_model(model, docs, doc_ids)
    history = [{"epoch": 0, **e0}]
    best = (-1.0, 0)    # best *fine-tuned* epoch; epoch 0 (base model) is logged for reference only
    print(f"epoch 0: train nDCG={e0['train']['ndcg10']:.4f}  val {e0['val']}")
    for ep in range(1, EPOCHS + 1):
        model.max_seq_length = MAX_LEN
        model.fit(train_objectives=[(loader, loss)], epochs=1, warmup_steps=int(0.1 * len(loader)) if ep == 1
                  else 0, optimizer_params={"lr": LR}, show_progress_bar=True, use_amp=DEVICE == "cuda")
        model.max_seq_length = 512
        e = evaluate_model(model, docs, doc_ids)
        mv, mt = e["val"], e["train"]
        history.append({"epoch": ep, **e})
        print(f"epoch {ep}: train nDCG={mt['ndcg10']:.4f}  val {mv}")
        if mv["ndcg10"] > best[0]:
            best = (mv["ndcg10"], ep)
            model.save(out_dir)
    with open(os.path.join(CACHE_DIR, f"ft_{mode}_history.json"), "w") as f:
        json.dump({"history": history, "best_epoch": best[1]}, f, indent=2)
    return out_dir, history, best


def main():
    from task2_dense import run_dense
    modes = sys.argv[1:] or ["random", "hard"]
    summary = {}
    for mode in modes:
        out_dir, history, best = train(mode)
        m, cost = run_dense(out_dir, f"ft_{mode}", trec_eval=f"t4_{mode}.trec")
        log_metrics(f"t4_{mode}", m)
        summary[f"val_ndcg10_{mode}_negatives"] = round(m["ndcg10"], 4)
        summary[f"{mode}_best_epoch"] = best[1]
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    summary["base_model"] = (f"{BASE} fine-tuned with MultipleNegativesRankingLoss (in-batch + 1 explicit "
                             f"negative), lr={LR}, batch={BATCH}, max_len={MAX_LEN}, up to {EPOCHS} epochs, "
                             f"best epoch by val nDCG@10")
    update_results("task4", summary)


if __name__ == "__main__":
    main()
