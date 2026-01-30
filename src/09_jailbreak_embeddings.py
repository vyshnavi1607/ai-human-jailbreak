import pickle
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def main():
    print("Loading preprocessed jailbreak dataset...")
    ds = load_from_disk("data/jailbreak_clean")

    texts = ds["text"]
    labels = ds["label"]

    print(f"Total samples: {len(texts)}")

    print("Loading NOMIC embedding model (CPU)...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1",
        device="cpu",
        trust_remote_code=True
    )

    embeddings = []
    batch_size = 8   # SAFE for CPU

    print("Generating embeddings...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        emb = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings.append(emb)

    X = np.vstack(embeddings)
    y = np.array(labels)

    print("Saving embeddings...")
    with open("data/embeddings/X_jailbreak.pkl", "wb") as f:
        pickle.dump(X, f)

    with open("data/embeddings/y_jailbreak.pkl", "wb") as f:
        pickle.dump(y, f)

    print("Jailbreak embeddings generated and saved successfully ✅")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    main()
