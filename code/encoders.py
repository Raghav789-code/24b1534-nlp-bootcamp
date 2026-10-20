"""Bi-encoder wrappers with a common interface: encode_queries / encode_docs -> L2-normalised float32."""
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class STEncoder:
    """Any sentence-transformers model (optionally with a query instruction prefix)."""

    def __init__(self, name, query_prefix="", max_len=512):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name, device=DEVICE)
        self.model.max_seq_length = max_len
        if DEVICE == "cuda":
            self.model.half()     # fp16 inference: half the memory, ~2x faster
        self.query_prefix = query_prefix

    def encode_queries(self, qs, bs=64):
        return self.model.encode([self.query_prefix + q for q in qs], batch_size=bs,
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def encode_docs(self, docs, bs=16):
        texts = [d["title"] + ". " + " ".join(d["abstract"]) for d in docs]
        # sort by length so batches are padded less
        order = np.argsort([-len(t) for t in texts])
        emb = self.model.encode([texts[i] for i in order], batch_size=bs, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        out = np.empty_like(emb)
        out[order] = emb
        return out

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())


class MedCPTEncoder:
    """NCBI MedCPT: separate query / article encoders (each ~110M), CLS pooling, inner product."""

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.name = "ncbi/MedCPT-Query-Encoder + ncbi/MedCPT-Article-Encoder"
        self.qtok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.qenc = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(DEVICE).eval()
        self.dtok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self.denc = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(DEVICE).eval()
        if DEVICE == "cuda":
            self.qenc.half()
            self.denc.half()

    @torch.no_grad()
    def _run(self, tok, enc, batch, max_len):
        x = tok(*batch, truncation=True, padding=True, return_tensors="pt", max_length=max_len).to(DEVICE)
        return enc(**x).last_hidden_state[:, 0, :].float().cpu().numpy()

    def encode_queries(self, qs, bs=64):
        out = [self._run(self.qtok, self.qenc, (qs[i:i + bs],), 64) for i in range(0, len(qs), bs)]
        return np.concatenate(out).astype(np.float32)

    def encode_docs(self, docs, bs=16):
        from tqdm import tqdm
        order = np.argsort([-len(" ".join(d["abstract"])) for d in docs])
        embs = []
        for i in tqdm(range(0, len(docs), bs), desc="MedCPT docs"):
            b = [docs[j] for j in order[i:i + bs]]
            embs.append(self._run(self.dtok, self.denc,
                                  ([d["title"] for d in b], [" ".join(d["abstract"]) for d in b]), 512))
        emb = np.concatenate(embs).astype(np.float32)
        out = np.empty_like(emb)
        out[order] = emb
        return out

    def n_params(self):
        return sum(p.numel() for m in (self.qenc, self.denc) for p in m.parameters())


BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def get_encoder(kind):
    if kind == "general":
        return STEncoder("BAAI/bge-base-en-v1.5", query_prefix=BGE_PREFIX)
    if kind == "scientific":
        return MedCPTEncoder()
    return STEncoder(kind, query_prefix=BGE_PREFIX)   # path to a fine-tuned bge checkpoint
