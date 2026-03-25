# CSL 7640 Assignment 2: Natural Language Understanding

**Author:** Prashant Kumar (Roll No: M25CSE023)  
**Instructor:** Anand Mishra  
**Institution:** Indian Institute of Technology Jodhpur  

This repository contains the implementation and analysis for two independent NLP problems:
1.  **Problem 1:** Training Word2Vec models (CBOW and Skip-gram) from scratch and using Gensim on a custom corpus scraped from the IIT Jodhpur website and academic PDFs, followed by semantic analysis and dimensionality reduction visualizations.
2.  **Problem 2:** Implementing and training three character-level sequence models (Vanilla RNN, Bidirectional LSTM, and Attention-augmented RNN) to generate plausible Indian names.

## Setup Instructions

Tested on Windows 11 with Python 3.11.

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
    *(Note: If you encounter a script execution error on Windows, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first).*

2.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *(Note: The requirements install the CPU-only build of PyTorch).*

## Execution Guide

### Problem 1: Word Embeddings
Navigate to the `problem1` directory. Tasks must be run sequentially as each script generates files required by the next.

* **Task 1 (Data Collection):** Scrapes the web pages and extracts PDFs to build the cleaned corpus. *Takes 15–30 minutes depending on internet speed*.
    ```bash
    python task1_corpus.py
    ```
* **Task 2 (Model Training):** Trains the CBOW and Skip-gram models (both Gensim and from-scratch implementations).
    ```bash
    python task2_train.py
    ```
* **Task 3 (Semantic Analysis):** Runs nearest neighbors and analogy computations.
    ```bash
    python task3_analysis.py
    ```
* **Task 4 (Visualization):** Generates PCA, t-SNE, and training loss plots.
    ```bash
    python task4_visualize.py
    ```

### Problem 2: Character-Level Name Generation
Navigate to the `problem2` directory.

* **Task 0 (Dataset Generation):** Generates the `TrainingNames.txt` dataset programmatically.
    ```bash
    python generate_names.py
    ```
* **Task 2 (Training):** Trains the Vanilla RNN, BLSTM, and Attention RNN models. *Expect 5–20 minutes on CPU*.
    ```bash
    python train.py
    ```
* **Task 3 (Generation):** Generates sample names using the trained checkpoints.
    ```bash
    python generate.py
    ```
* **Task 4 (Evaluation):** Evaluates novelty and diversity, and produces length and metric plots.
    ```bash
    python evaluate.py
    ```