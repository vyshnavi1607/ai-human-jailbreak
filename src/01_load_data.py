import os
from datasets import load_dataset

print("Script started...")

dataset = load_dataset("silentone0725/ai-human-text-detection-v1")

print("Dataset loaded")

save_path = "data/ai_human"
os.makedirs(save_path, exist_ok=True)

dataset.save_to_disk(save_path)

print("Dataset saved successfully")
