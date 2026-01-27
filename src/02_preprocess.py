"""
02_preprocess.py

Purpose:
- Load AI vs Human dataset from disk
- Apply minimal text cleaning
- Save cleaned dataset back to disk
"""

import re
from datasets import load_from_disk


def clean_text(example):
    """
    Minimal preprocessing:
    - Convert to string
    - Remove extra whitespace
    """
    text = str(example["text"])
    text = re.sub(r"\s+", " ", text)
    example["text"] = text.strip()
    return example


def main():
    print("Loading dataset from disk...")
    dataset = load_from_disk("data/ai_human_raw")

    print("Applying minimal preprocessing...")
    dataset = dataset.map(clean_text)

    print("Saving preprocessed dataset...")
    dataset.save_to_disk("data/ai_human_clean")

    print("Preprocessing completed successfully ✅")


if __name__ == "__main__":
    main()
