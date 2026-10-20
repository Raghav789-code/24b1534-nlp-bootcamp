"""Task 1 - corpus exploration + BM25 (own sparse implementation), tuned on validation nDCG@10.

Outputs: predictions/t1_bm25.trec (eval), predictions/val_bm25.trec (val),
         cache/bm25_{val,eval,train}.npy (full score matrices for later tasks)
"""
import itertools
import os
import re
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp

from common import (CACHE_DIR, PRED_DIR, evaluate, load_claims, load_corpus, log_metrics, qrels_of,
                    save_run, set_seed, topk_from_scores, update_results, write_trec)

try:
    import Stemmer
    _stemmer = Stemmer.Stemmer("english")
    stem_words = _stemmer.stemWords
except ImportError:  # fallback
    from nltk.stem import PorterStemmer
    _ps = PorterStemmer()
    stem_words = lambda ws: [_ps.stem(w) for w in ws]  # noqa: E731

STOP = set("""a an and are as at be but by for if in into is it no not of on or such that the their then there
these they this to was will with from we our were which been has have had its than also both can may using
""".split())
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text, stem=True, stop=True):
    toks = TOKEN_RE.findall(text.lower())
    if stop:
        toks = [t for t in toks if t not in STOP]
    return stem_words(toks) if stem else toks


class BM25:
    def __init__(self, texts, stem=True, stop=True):
        self.stem, self.stop = stem, stop
        toks = [tokenize(t, stem, stop) for t in texts]
        self.vocab = {}
        rows, cols, vals = [], [], []
        for i, ts in enumerate(toks):
            for w, c in Counter(ts).items():
                j = self.vocab.setdefault(w, len(self.vocab))
                rows.append(i); cols.append(j); vals.append(c)
        self.tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(self.vocab)), dtype=np.float32)
        self.dl = np.asarray(self.tf.sum(1)).ravel()
        self.avgdl = self.dl.mean()
        df = np.bincount(self.tf.indices, minlength=len(self.vocab))
        n = len(texts)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.W = None

    def fit(self, k1=1.2, b=0.75):
        tf = self.tf.tocoo()
        norm = k1 * (1 - b + b * self.dl / self.avgdl)
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.W = sp.csr_matrix((w, (tf.row, tf.col)), shape=self.tf.shape).T.tocsr()  # terms x docs
        return self

    def query_matrix(self, queries):
        rows, cols, vals = [], [], []
        for i, q in enumerate(queries):
            for w, c in Counter(tokenize(q, self.stem, self.stop)).items():
                if w in self.vocab:
                    rows.append(i); cols.append(self.vocab[w]); vals.append(c)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(queries), len(self.vocab)), dtype=np.float32)

    def score(self, queries):
        return np.asarray((self.query_matrix(queries) @ self.W).todense())


def explore(docs, splits):
    lens = [len(" ".join(d["abstract"]).split()) for d in docs]
    print(f"corpus: {len(docs)} abstracts, words/abstract mean={np.mean(lens):.0f} "
          f"median={np.median(lens):.0f} max={max(lens)}; sentences mean="
          f"{np.mean([len(d['abstract']) for d in docs]):.1f}")
    for name, cl in splits.items():
        if "evidence" not in cl[0]:
            print(f"{name}: {len(cl)} claims (unlabelled)")
            continue
        nei = sum(1 for c in cl if not c["evidence"])
        labs = Counter(e["label"] for c in cl for e in c["evidence"])
        ndocs = Counter(len(c["evidence"]) for c in cl)
        print(f"{name}: {len(cl)} claims, NEI={nei}, labels={dict(labs)}, #gold docs/claim={dict(ndocs)}")


def main():
    set_seed()
    docs, doc_ids, texts = load_corpus()
    splits = {s: load_claims(s) for s in ["train", "val", "eval"]}
    explore(docs, splits)
    val, qrels = splits["val"], qrels_of(splits["val"])
    vq, vids = [c["claim"] for c in val], [int(c["id"]) for c in val]

    best = None
    for stem, stop in [(True, True), (True, False), (False, True)]:
        bm = BM25(texts, stem, stop)
        for k1, b in itertools.product([0.6, 0.9, 1.2, 1.5, 2.0], [0.3, 0.5, 0.75, 0.9, 1.0]):
            S = bm.fit(k1, b).score(vq)
            run = dict(zip(vids, topk_from_scores(S, doc_ids)))
            m = evaluate(run, qrels)
            if best is None or m["ndcg10"] > best[0]["ndcg10"]:
                best = (m, dict(stem=stem, stop=stop, k1=k1, b=b))
    m, cfg = best
    print("best BM25 config:", cfg)
    log_metrics("t1_bm25", m)

    bm = BM25(texts, cfg["stem"], cfg["stop"]).fit(cfg["k1"], cfg["b"])
    t0 = time.time()
    for split in ["val", "eval", "train"]:
        cl = splits[split]
        ids = [int(c["id"]) for c in cl]
        S = bm.score([c["claim"] for c in cl])
        np.save(os.path.join(CACHE_DIR, f"bm25_{split}.npy"), S.astype(np.float32))
        ranked = topk_from_scores(S, doc_ids)
        save_run(f"bm25_{split}", ids, ranked)
        if split == "eval":
            write_trec(os.path.join(PRED_DIR, "t1_bm25.trec"), ids, ranked, "bm25")
        elif split == "val":
            write_trec(os.path.join(PRED_DIR, "val_bm25.trec"), ids, ranked, "bm25")
    print(f"BM25 scoring of all splits took {time.time() - t0:.2f}s")
    settings = (f"own sparse BM25 (Lucene idf), k1={cfg['k1']}, b={cfg['b']}, title+abstract, "
                f"lowercase regex tokens, stopwords={'removed' if cfg['stop'] else 'kept'}, "
                f"{'Snowball stemming' if cfg['stem'] else 'no stemming'}; grid-searched on val nDCG@10")
    update_results("task1", {"bm25_settings": settings, "val_ndcg10": round(m["ndcg10"], 4)})


if __name__ == "__main__":
    main()
