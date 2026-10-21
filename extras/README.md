# extras/ (not part of the graded submission)

The repo root follows the required submission layout exactly: `code/`, `predictions/`, `results.json`,
`report.md`. Everything else is kept in this folder so nothing is lost.

| Path | What it is |
|---|---|
| `24b1534_nlp_bootcamp.zip` | **The file to submit.** It packs the four root items and passes `check_format.py` ("All good.") |
| `data/` | The provided dataset (corpus, train/val/eval claims) |
| `NLP_PS.pdf`, `check_format.py`, `results_template.json` | The provided problem statement and helper files |
| `cache/` | Intermediate outputs: score matrices, runs, cross-encoder and NLI scores, logs, and the two fine-tuned BGE models (`cache/ft_random`, `cache/ft_hard`, stored with Git LFS) |

## Validation results

| Run | nDCG@10 |
|---|---|
| BM25 | 0.855 |
| BGE-base (general) | 0.891 |
| MedCPT (biomedical) | 0.851 |
| Hybrid, weighted RRF | 0.914 |
| **MedCPT cross-encoder rerank (k=20)** | **0.928** |
| Fine-tuned BGE, random negatives | 0.887 |
| Fine-tuned BGE, hard negatives | 0.892 |

Claim verification on validation: F1 = 0.539 (P 0.621, R 0.476).

## Reproduce (from the repo root)

```bash
pip install -r code/requirements.txt
cd code
python task1_bm25.py && python task2_dense.py && python task3_hybrid_rerank.py
python task4_finetune.py && python task5_verify.py && python report_analysis.py
python ../extras/check_format.py ..
```

The scripts find `extras/data/` and write to `extras/cache/` automatically. To get the large model files,
install Git LFS (`git lfs install`) before cloning.
