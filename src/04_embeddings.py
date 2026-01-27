"""
04_embeddings.py

Purpose:
- Generate NOMIC embeddings for AI vs Human dataset
- CPU-friendly, batch-wise processing
- Save embeddings to disk
"""

import os
import pickle
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer


def main():
    print("Loading encoded dataset...")
    dataset = load_from_disk("data/ai_human_encoded")

    texts = dataset["train"]["text"]
    labels = dataset["train"]["label"]

    print(f"Number of training samples: {len(texts)}")

    print("Loading NOMIC embedding model (CPU)...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1",
        trust_remote_code=True,
        device="cpu"
    )

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True
    )

    os.makedirs("data/embeddings", exist_ok=True)

    with open("data/embeddings/X_train.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    with open("data/embeddings/y_train.pkl", "wb") as f:
        pickle.dump(labels, f)

    print("Embeddings generated and saved successfully ✅")


if __name__ == "__main__":
    main()

