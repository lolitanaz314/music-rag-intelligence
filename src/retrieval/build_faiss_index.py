import argparse
from pathlib import Path
import numpy as np
import faiss


def main(embeddings_path: str, out_index_path: str, normalize: bool = True) -> None:
    emb = np.load(embeddings_path).astype(np.float32, copy=False)

    # Cosine similarity via inner product on normalized vectors
    if normalize:
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
    else:
        index = faiss.IndexFlatL2(emb.shape[1])

    index.add(emb)

    out = Path(out_index_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))

    print("[DONE] FAISS index built")
    print("Index path:", out)
    print("Vectors:", index.ntotal)
    print("Dim:", emb.shape[1])
    print("Metric:", "cosine(IP)" if normalize else "L2")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", default="data/merged/all_embeddings.npy")
    p.add_argument("--out", default="data/vectorstore/faiss.index")
    p.add_argument("--normalize", action="store_true", help="Use cosine similarity (recommended)")
    args = p.parse_args()

    main(args.embeddings, args.out, normalize=args.normalize)
