"""
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder


def main():
    print("Loading jailbreak dataset...")
    ds = load_dataset("BallAdMyFi/jailbreaking_prompt_v2")["train"]

    print("Initial samples:", len(ds))

    # Minimal cleaning (DO NOT over-clean)
    def clean(example):
    text = example.get("text")

    if text is None:
        example["text"] = ""
        return example

    text = text.strip()
    text = " ".join(text.split())
    example["text"] = text
    return example

    def clean(example):
    text = example.get("text")

    if text is None:
        example["text"] = ""
        return example

    text = text.strip()
    text = " ".join(text.split())
    example["text"] = text
    return example

        text = example["text"]
        text = text.strip()
        text = " ".join(text.split())
        example["text"] = text
        return example

    print("Applying minimal preprocessing...")
    ds = ds.map(clean)

    # Encode labels: safe=0, unsafe=1
    print("Encoding labels...")
    le = LabelEncoder()
    ds = ds.map(lambda x: {"label": le.fit_transform([x["label"]])[0]})

    print("Label mapping:")
    print(dict(zip(le.classes_, le.transform(le.classes_))))

    # Save processed dataset
    save_path = "data/jailbreak_clean"
    ds.save_to_disk(save_path)

    print(f"Preprocessed dataset saved to {save_path}")


if __name__ == "__main__":
    main()

"""


from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder


def main():
    print("Loading jailbreak dataset...")
    ds = load_dataset("BallAdMyFi/jailbreaking_prompt_v2")["train"]

    print("Initial samples:", len(ds))

    # -----------------------------
    # Minimal & SAFE preprocessing
    # -----------------------------
    def clean(example):
        text = example.get("text")

        # Handle None values safely
        if text is None:
            example["text"] = ""
            return example

        # Minimal normalization only
        text = text.strip()
        text = " ".join(text.split())
        example["text"] = text
        return example

    print("Applying minimal preprocessing...")
    ds = ds.map(clean)

    # -----------------------------
    # Label Encoding
    # safe   -> 0
    # unsafe -> 1
    # -----------------------------
    print("Encoding labels...")
    le = LabelEncoder()

    labels = ds["label"]
    le.fit(labels)

    ds = ds.map(lambda x: {"label": le.transform([x["label"]])[0]})

    print("Label mapping:")
    for cls, val in zip(le.classes_, le.transform(le.classes_)):
        print(f"  {cls} -> {val}")

    # -----------------------------
    # Save processed dataset
    # -----------------------------
    save_path = "data/jailbreak_clean"
    ds.save_to_disk(save_path)

    print(f"\nPreprocessed dataset saved successfully at: {save_path}")
    print("Total samples after preprocessing:", len(ds))


if __name__ == "__main__":
    main()
