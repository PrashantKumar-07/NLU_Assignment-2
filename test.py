import sys
import os
import json
import pickle

# 1. Get the current directory of test.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Add 'problem1' to the system path so Python can find 'vocab.py'
problem1_dir = os.path.join(base_dir, "problem1")
sys.path.append(problem1_dir)

# 3. Set the exact absolute paths to your files
model_path = os.path.join(problem1_dir, "models", "scratch_skipgram_d200_w5_n10.pkl")
vocab_path = os.path.join(problem1_dir, "models", "vocabulary.pkl")
stats_path = os.path.join(problem1_dir, "outputs", "corpus_stats.json")

print("="*60)
print("ANSWER FOR P1: 200-Dimensional Vector for 'student'")
print("="*60)

try:
    # Load vocabulary
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        
    # Load the 200-dim model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    idx = vocab.word2idx["student"]
    vec = model["W_in"][idx]
    
    # Format the vector
    vector_str = ", ".join([f"{x:.4f}" for x in vec])
    print(f"student - {vector_str}")
    
except Exception as e:
    print(f"Error loading model/vocab: {e}\nCheck if the files exist at {model_path}")

print("\n" + "="*60)
print("ANSWER FOR P1: Top-10 words (frequency-wise)")
print("="*60)

try:
    # Load stats JSON
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    # Extract using the correct key from your task1_corpus.py
    top_10 = stats["top_50_words"][:10] 
    
    # Format the output
    freq_str = ", ".join([f"{w}, {c}" for w, c in top_10])
    print(freq_str)
    
except Exception as e:
    print(f"Error loading stats: {e}\nCheck if the file exists at {stats_path}")