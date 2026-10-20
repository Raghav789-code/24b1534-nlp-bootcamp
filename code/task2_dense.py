"""Task 2 - dense retrieval with a general (BGE-base) and a biomedical (MedCPT) bi-encoder.

Outputs: predictions/t2_general.trec, t2_scientific.trec (eval), val_general.trec (val),
         cache/{tag}_{split}.npy score matrices, cache/{tag}_docemb.npy
"""
import os
import sys
import time

import numpy as np
import torch

from common import (CACHE_DIR, PRED_DIR, evaluate, load_claims, load_corpus, log_metrics, qrels_of,
                    save_run, set_seed, topk_from_scores, update_results, write_trec)
from encoders import DEVICE, get_encoder


def run_dense(kind, tag, trec_eval=None, trec_val=None):
    """Encode corpus + all claim splits with encoder `kind`; returns (val metrics, cost stats)."""
    set_seed()
    docs, doc_ids, _ = load_corpus()
    enc = get_encoder(kind)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    D = enc.encode_docs(docs)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    index_time = time.time() - t0
    np.save(os.path.join(CACHE_DIR, f"{tag}_docemb.npy"), D)

    metrics = None
    for split in ["val", "eval", "train"]:
        cl = load_claims(split)
        ids = [int(c["id"]) for c in cl]
        S = enc.encode_queries([c["claim"] for c in cl]) @ D.T
        np.save(os.path.join(CACHE_DIR, f"{tag}_{split}.npy"), S.astype(np.float32))
        ranked = topk_from_scores(S, doc_ids)
        save_run(f"{tag}_{split}", ids, ranked)
        if split == "val":
            metrics = evaluate(dict(zip(ids, ranked)), qrels_of(cl))
            log_metrics(tag, metrics)
            if trec_val:
                write_trec(os.path.join(PRED_DIR, trec_val), ids, ranked, tag)
        if split == "eval" and trec_eval:
            write_trec(os.path.join(PRED_DIR, trec_eval), ids, ranked, tag)

    # online query latency: one claim at a time, encode + exact search + top-100
    qs = [c["claim"] for c in load_claims("val")]
    enc.encode_queries(qs[:5], bs=1)  # warm-up
    t0 = time.time()
    for q in qs:
        s = enc.encode_queries([q], bs=1) @ D.T
        topk_from_scores(s, doc_ids)
    latency_ms = (time.time() - t0) / len(qs) * 1000
    wbytes = 2 if DEVICE == "cuda" else 4
    memory_mb = (enc.n_params() * wbytes + D.nbytes) / 2 ** 20   # model weights (fp16 on GPU) + flat fp32 index
    cost = {"index_time_sec": round(index_time, 2), "query_latency_ms": round(latency_ms, 2),
            "memory_mb": round(memory_mb, 1), "index_mb": round(D.nbytes / 2 ** 20, 1)}
    print(f"{tag}: {cost}")
    del enc
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return metrics, cost


def main():
    names = {"general": "BAAI/bge-base-en-v1.5 (109M, query instruction prefix, max 512 tokens)",
             "scientific": "ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder (110M each, "
                           "PubMed-trained, CLS + inner product)"}
    out = {"general_model": names["general"], "scientific_model": names["scientific"],
           "index_time_sec": {}, "query_latency_ms": {}, "memory_mb": {}}
    kinds = sys.argv[1:] or ["general", "scientific"]
    for kind in kinds:
        m, cost = run_dense(kind, kind, trec_eval=f"t2_{kind}.trec",
                            trec_val="val_general.trec" if kind == "general" else None)
        out[f"val_ndcg10_{kind}"] = round(m["ndcg10"], 4)
        for k in ["index_time_sec", "query_latency_ms", "memory_mb"]:
            out[k][kind] = cost[k]
    import json
    from common import RESULTS_PATH
    with open(RESULTS_PATH) as f:
        prev = json.load(f)["task2"]
    for k in ["index_time_sec", "query_latency_ms", "memory_mb"]:
        prev[k].update(out.pop(k))
    prev.update(out)
    update_results("task2", prev)


if __name__ == "__main__":
    main()
