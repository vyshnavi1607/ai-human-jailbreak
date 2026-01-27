# AI vs Human Text Detection with Jailbreak Analysis

## Project Overview

This project proposes a modular and hierarchical framework for analyzing text behavior in the context of AI safety. The system separates authorship detection (AI vs Human) from intent detection (Safe vs Jailbreak), treating them as independent but connected classification tasks.

## Motivation

With the rise of large language models, it has become increasingly difficult to:

- Distinguish AI-generated text from human-written text
- Detect malicious or jailbreaking prompts intended to bypass AI safety mechanisms

This project addresses these challenges using semantic embeddings and classical machine learning classifiers.

## Project Structure

ai-human-jailbreak/
├── src/
│ ├── 01_load_data.py
│ ├── 02_preprocess.py
│ ├── 03_label_encode.py
│ ├── 04_embeddings.py
│ ├── 05_train_model.py
│ └── 06_inference.py
│
├── data/ # ignored by git (datasets, embeddings)
├── venv/ # virtual environment
├── .gitignore
└── README.md

## Methodology

### Stage 1: AI vs Human Classification

- Dataset: silentone0725/ai-human-text-detection-v1
- Task: Binary classification (AI vs Human)
- Embedding: nomic-ai/nomic-embed-text-v1
- Classifiers: Logistic Regression, Random Forest, XGBoost (comparative)

### Stage 2: Jailbreak Detection (Future Work)

- Dataset: BallAdMyFi/jailbreaking_prompt_v2
- Task: Binary classification (Safe vs Jailbreak)
- Uses the same embedding and classification pipeline

## Key Design Decisions

- Hierarchical binary classification instead of multi-class
- Transformer-based embeddings with classical ML classifiers
- Clear separation between authorship and intent

## Status

- Stage 1 pipeline implemented
- Embedding experiments in progress
- Model training and evaluation next
