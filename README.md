# Inter-IIT Tech Meet 15.0 – NLP Selection Bootcamp: SciFact

Evidence retrieval and claim verification on SciFact. Roll no. 24b1534.

**The file to submit is [`24b1534_nlp_bootcamp.zip`](24b1534_nlp_bootcamp.zip).** It contains `code/`,
`predictions/`, `results.json` and `report.md` and passes `check_format.py` ("All good.").

| Path | What it is |
|---|---|
| `code/` | All scripts. [`code/README.md`](code/README.md) gives the exact command for each task and the validation results |
| `code/colab_pipeline.ipynb` | The whole pipeline in one Colab notebook (upload the data, then Run all) |
| `predictions/` | The 7 eval `.trec` runs, the 2 val runs and the 3 Task 5 CSVs |
| `results.json`, `report.md` | Filled results template and answers to Q1–Q5 |
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

## Reproduce

```bash
pip install -r code/requirements.txt
cd code
python task1_bm25.py && python task2_dense.py && python task3_hybrid_rerank.py
python task4_finetune.py && python task5_verify.py && python report_analysis.py
python ../check_format.py ..
```

To get the large model files, install Git LFS (`git lfs install`) before cloning.
