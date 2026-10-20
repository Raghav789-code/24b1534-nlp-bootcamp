# SciFact evidence retrieval + claim verification — reproduction

All scripts live in `code/` and are run **from inside `code/`**. They expect the provided data in
`../../data/` (i.e. the `data/` folder next to `check_format.py`); override with the `DATA_DIR`
environment variable. Global seed = 42 (`common.SEED`). Hardware used: 1x RTX 4050 Laptop (6 GB),
Python 3.13, torch 2.13 + CUDA. Intermediate score matrices / runs go to `../cache/` (not shipped).

```bash
pip install -r requirements.txt
```

Run the tasks in order (each one reuses the cached runs of the previous ones and writes its numbers
into `../results.json` and `val_metrics.json`):

| Task | Command | Output files (`../predictions/`) | Approx. time (GPU) |
|---|---|---|---|
| 1 BM25 | `python task1_bm25.py` | `t1_bm25.trec`, `val_bm25.trec` | < 1 min |
| 2 Dense | `python task2_dense.py` | `t2_general.trec`, `t2_scientific.trec`, `val_general.trec` | ~5 min |
| 3 Hybrid + rerank | `python task3_hybrid_rerank.py` | `t3_hybrid.trec`, `t3_rerank.trec` | ~15 min |
| 4 Fine-tuning | `python task4_finetune.py` | `t4_random.trec`, `t4_hard.trec` | ~6 min |
| 5 Verification | `python task5_verify.py` | `t5_verification.csv`, `t5_val_verification.csv`, `t5_val_errors.csv` | ~5 min |
| Q1 analysis | `python report_analysis.py` | fills `report_q1` in `results.json`, prints the ablation | seconds |
| Report draft | `python make_report.py` | `../report.md` filled from the measured numbers | seconds |

Then validate: `python ../../check_format.py ..`

The submitted `report.md` starts from the `make_report.py` draft, with Q1's explanations written by hand from
`report_analysis.py`'s output. `colab_pipeline.ipynb` holds the same code in one notebook: upload the data zip, then
Runtime → Run all.

## Validation results (seed 42)

| Run | nDCG@10 | Recall@100 | MRR |
|---|---|---|---|
| t1 BM25 (k1=0.9, b=0.5) | 0.855 | 0.981 | 0.828 |
| t2 general (BGE-base) | 0.891 | 1.000 | 0.879 |
| t2 scientific (MedCPT) | 0.851 | 1.000 | 0.813 |
| t3 hybrid (weighted RRF) | 0.914 | 1.000 | 0.896 |
| t3 rerank (MedCPT cross-encoder, k=20) | **0.928** | 1.000 | 0.907 |
| t4 random negatives | 0.887 | 1.000 | 0.872 |
| t4 hard negatives | 0.892 | 1.000 | 0.877 |

Task 5 (top-2 reranked abstracts): validation P = 0.621, R = 0.476, F1 = 0.539.

## What each script does

* `common.py` – data loading, TREC writing (100 unique docs per claim, non-increasing scores),
  nDCG@10 / Recall@100 / MRR computed only over claims with gold evidence (as in grading).
* `task1_bm25.py` – corpus/claim statistics, own sparse-matrix BM25 (Lucene idf) over title+abstract;
  grid search on val nDCG@10 over k1, b, stemming and stop-word removal.
* `task2_dense.py` / `encoders.py` – general model `BAAI/bge-base-en-v1.5` (with its query instruction)
  and biomedical model `ncbi/MedCPT` (query encoder + article encoder). Exact inner-product search.
  Index time = encoding the 5,183 abstracts; query latency = encode one claim + search + top-100, mean
  over the 162 val claims, batch size 1; memory = model weights (fp16 on GPU) + fp32 flat index.
* `task3_hybrid_rerank.py` – fuses full-corpus scores of BM25, BGE and MedCPT (z-score, min-max or
  RRF). Fusion weights are grid-searched on the **train** claims (all three models are pretrained only,
  so train is unseen by them); normalisation and reranker depth k are chosen on val nDCG@10.
  The best first-stage system (by val) is reranked with a cross-encoder. `BAAI/bge-reranker-base`,
  `cross-encoder/ms-marco-MiniLM-L-12-v2` and `ncbi/MedCPT-Cross-Encoder` are compared on val with
  k ∈ {5, 10, 20, 30, 50, 100}. The top-k go in reranker order; ranks k+1..100 keep first-stage order.
* `task4_finetune.py` – fine-tunes `bge-base-en-v1.5` on the train claims (claim → gold abstract) with
  MultipleNegativesRankingLoss, (a) one random negative per pair, (b) one hard negative per pair mined
  from the top-30 of our BM25 / BGE / MedCPT train runs. lr 1e-5, batch 16, 2 epochs. A first run at
  lr 2e-5 overfit after one epoch. Val and train nDCG@10 are logged every epoch
  (`../cache/ft_*_history.json`), and the best fine-tuned epoch on val is kept.
* `task5_verify.py` – takes the top abstracts of the retriever with the best val nDCG@10, runs zero-shot
  NLI (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`) per abstract sentence and on the whole abstract,
  aggregates the probabilities with a 3-way logistic regression trained on the train claims
  (decision margin + number of abstracts tuned with 5-fold out-of-fold predictions on train), and writes
  the eval and val predictions. Every val claim whose predicted (doc, label) set differs from gold is
  tagged with one error category by deterministic rules (retrieval_miss → needs_multiple_docs →
  negation / numerical cue words → entity_mismatch for wrong extra abstracts → other).
* `report_analysis.py` – per-claim val nDCG@10 BM25 vs BGE for Q1 and the ablation table.

All models are open-source encoders with ≤ 400M parameters; no generative LLMs are used.
