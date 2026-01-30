

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
import pickle


def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1",
        device="cpu",
        trust_remote_code=True
    )

    print("Loading precomputed embeddings (for demo)...")
    with open("data/embeddings/X_all.pkl", "rb") as f:
        X_ai = pickle.load(f)
    with open("data/embeddings/y_all.pkl", "rb") as f:
        y_ai = pickle.load(f)

    with open("data/embeddings/X_jailbreak.pkl", "rb") as f:
        X_jb = pickle.load(f)
    with open("data/embeddings/y_jailbreak.pkl", "rb") as f:
        y_jb = pickle.load(f)

    print("Training lightweight demo models (one-time)...")
    ai_model = SVC(kernel="poly")
    ai_model.fit(X_ai[:5000], y_ai[:5000])  # small subset = FAST

    jb_model = SVC(kernel="poly")
    jb_model.fit(X_jb, y_jb)

    print("\n=== DEMO READY ===")
    print("Type text and press Enter (type 'exit' to quit)\n")

    while True:
        text = input("Input: ")
        if text.lower() == "exit":
            break

        emb = embedder.encode([text], normalize_embeddings=True)

        ai_pred = ai_model.predict(emb)[0]
        jb_pred = jb_model.predict(emb)[0]

        print("\nPrediction:")
        print("AI vs Human :", "AI-generated" if ai_pred == 1 else "Human-written")
        print("Jailbreak   :", "Unsafe / Jailbreak" if jb_pred == 1 else "Safe")
        print("-" * 40)


if __name__ == "__main__":
    main()
