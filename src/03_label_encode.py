"""
03_label_encode.py

Purpose:
- Encode labels (ai / human) into numeric values
- Save encoded dataset
- Save label mapping for future inference
"""

import os
import pickle
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder


def main():
    print("Loading preprocessed dataset...")
    dataset = load_from_disk("data/ai_human_clean")

    # Fit encoder on training labels only
    encoder = LabelEncoder()
    encoder.fit(dataset["train"]["label"])

    print("Label mapping:")
    for label, value in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        print(f"{label} → {value}")

    # Encode labels in all splits
    def encode_labels(example):
        example["label"] = int(encoder.transform([example["label"]])[0])
        return example

    print("Encoding labels...")
    encoded_dataset = dataset.map(encode_labels)

    # Create output directory
    save_path = "data/ai_human_encoded"
    os.makedirs(save_path, exist_ok=True)

    print("Saving encoded dataset...")
    encoded_dataset.save_to_disk(save_path)

    # Save label encoder
    with open("data/label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print("Label encoding completed successfully ✅")


if __name__ == "__main__":
    main()
