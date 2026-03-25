# CSL 7640 Assignment 2: Natural Language Understanding

[cite_start]**Author:** Prashant Kumar (Roll No: M25CSE023) [cite: 1314]
[cite_start]**Instructor:** Anand Mishra [cite: 1312]
[cite_start]**Institution:** Indian Institute of Technology Jodhpur [cite: 1315]

[cite_start]This repository contains the implementation and analysis for two independent NLP problems[cite: 1327]:
1.  [cite_start]**Problem 1:** Training Word2Vec models (CBOW and Skip-gram) from scratch and using Gensim on a custom corpus scraped from the IIT Jodhpur website and academic PDFs, followed by semantic analysis and dimensionality reduction visualizations[cite: 1328].
2.  [cite_start]**Problem 2:** Implementing and training three character-level sequence models (Vanilla RNN, Bidirectional LSTM, and Attention-augmented RNN) to generate plausible Indian names[cite: 1329].

## Setup Instructions

[cite_start]Tested on Windows 11 with Python 3.11[cite: 1669].

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
    [cite_start]*(Note: If you encounter a script execution error on Windows, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first)*[cite: 1670].

2.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    [cite_start]*(Note: The requirements install the CPU-only build of PyTorch)*[cite: 1671].

## Execution Guide

### Problem 1: Word Embeddings
[cite_start]Navigate to the `problem1` directory[cite: 1674]. [cite_start]Tasks must be run sequentially as each script generates files required by the next[cite: 1673].

* [cite_start]**Task 1 (Data Collection):** Scrapes the web pages and extracts PDFs to build the cleaned corpus[cite: 1674]. [cite_start]*Takes 15–30 minutes depending on internet speed*[cite: 1674].
    ```bash
    python task1_corpus.py
    ```
* [cite_start]**Task 2 (Model Training):** Trains the CBOW and Skip-gram models (both Gensim and from-scratch implementations)[cite: 1675].
    ```bash
    python task2_train.py
    ```
* [cite_start]**Task 3 (Semantic Analysis):** Runs nearest neighbors and analogy computations[cite: 1676].
    ```bash
    python task3_analysis.py
    ```
* [cite_start]**Task 4 (Visualization):** Generates PCA, t-SNE, and training loss plots[cite: 1676].
    ```bash
    python task4_visualize.py
    ```

### Problem 2: Character-Level Name Generation
[cite_start]Navigate to the `problem2` directory[cite: 1676].

* [cite_start]**Task 0 (Dataset Generation):** Generates the `TrainingNames.txt` dataset programmatically[cite: 1676].
    ```bash
    python generate_names.py
    ```
* [cite_start]**Task 2 (Training):** Trains the Vanilla RNN, BLSTM, and Attention RNN models[cite: 1676]. [cite_start]*Expect 5–20 minutes on CPU*[cite: 1676].
    ```bash
    python train.py
    ```
* [cite_start]**Task 3 (Generation):** Generates sample names using the trained checkpoints[cite: 1676].
    ```bash
    python generate.py
    ```
* [cite_start]**Task 4 (Evaluation):** Evaluates novelty and diversity, and produces length and metric plots[cite: 1676].
    ```bash
    python evaluate.py
    ```

---
Would you like me to review the directory structure to ensure everything is perfectly organized before you submit your assignment?