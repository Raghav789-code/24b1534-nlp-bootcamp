# Report (all analysis on validation claims)

Validation ablation (nDCG@10 / Recall@100 / MRR): BM25 0.855 / 0.981 / 0.828 · BGE-base 0.891 / 1.000 / 0.879 ·
MedCPT 0.851 / 1.000 / 0.813 · hybrid 0.914 / 1.000 / 0.896 · rerank 0.928 / 1.000 / 0.907 ·
FT-random 0.887 / 1.000 / 0.872 · FT-hard 0.892 / 1.000 / 0.877.

## Q1

BM25 beats BGE (`t2_general`) on claims **673** (nDCG@10 1.00 vs 0.36) and **381** (1.00 vs 0.50); BGE beats BM25 on
**766** (1.00 vs 0.00) and **1311** (1.00 vs 0.33). In 673, the rare exact string "LDL cholesterol" appears in the
gold title, while BGE prefers a broader cholesterol-history review. In 381, "environment" + "breast cancer" lexically
pins the gold study, and BGE ranks a topically similar meta-analysis first. In 766, "medications" paraphrases
"pharmacotherapy", so BM25 ranks gold 23rd. In 1311, "lncRNAs" and "LincRNA" share no token, but BGE links them.

## Q2

Hybrid fusion gave the largest gain. Weighted RRF over BM25, BGE and MedCPT raised val nDCG@10 from 0.891 (best single
retriever) to 0.914 (+0.023). Reranking the hybrid's top-20 with the MedCPT cross-encoder added another +0.014
(0.928). Fusion is nearly free: two dense query encodings (~35–48 ms each), BM25 (<1 ms) and a sum. Reranking costs
~430 ms per claim (20 cross-encoder passes of up to 512 tokens), about 10× the first-stage latency. General-domain
rerankers (bge-reranker-base, MiniLM-L12) lowered nDCG@10 at every k.

## Q3

Barely. Hard negatives (mined from the BM25/BGE/MedCPT top-30) reached 0.892, against 0.887 with random negatives.
The base BGE model scores 0.891, so with only 440 training pairs neither fine-tune clearly beats a strong pretrained
model. Overfitting was clear: we logged train and val nDCG@10 after every epoch. At lr 2e-5, train rose
0.909→0.924→0.929 while val fell 0.891→0.876→0.865. At lr 1e-5, val peaked after epoch 1 and then dropped (hard: train
0.930→0.936, val 0.892→0.885), so we kept the epoch-1 checkpoint.

## Q4

13 of 73 erroneous validation claims (**18%**) are `retrieval_miss`: a gold abstract fell outside the top-2 passed to
the verifier, even though Recall@100 is 1.0. The other 82% are verification errors: numerical 23, other 18,
entity_mismatch 10 (a label predicted for a non-gold abstract), negation 6 and needs_multiple_docs 3. Validation F1 is
0.539 (precision 0.621, recall 0.476). Categories come from deterministic rules applied in a fixed priority order (see
`code/README.md`).

## Q5

Fine-tune the NLI verifier on the SciFact training rationales instead of using it zero-shot. Train on (claim, gold
rationale sentences) → SUPPORT/CONTRADICT, and use retrieved non-rationale sentences as NEI. Retrieval is already
strong (nDCG@10 0.93, Recall@100 1.0), so most errors come from verification. MNLI-trained models miss
SciFact-specific contradictions (opposite effect direction, negated findings, numeric thresholds) and are poorly
calibrated about relevant-but-neutral abstracts. In-domain supervision targets exactly these failures.
