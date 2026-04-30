import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Loading saved Baseline and RAG+GRU models...")
if not os.path.exists('optimized_model.keras') or not os.path.exists('rag_artifacts.pkl') or not os.path.exists('baseline_model.keras'):
    print("\n[ERROR] Model files not found!")
    print("Please run `python main.py` first so it can train and save both models.")
    exit(1)

# Load both models
optimized = load_model('optimized_model.keras')
baseline = load_model('baseline_model.keras')

# Load the vectorizer, tokenizer, and RAG context data
with open('rag_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

vectorizer = artifacts['vectorizer']
tokenizer = artifacts['tokenizer']
doc_vectors = artifacts['doc_vectors']
texts_train = artifacts['texts_train']

MAX_LEN_BASE = 50
MAX_LEN_RAG = 200

def predict_both(text):
    # --- BASELINE PREDICTION ---
    # Tokenize and pad the raw text exactly like the baseline was trained
    seq_base = tokenizer.texts_to_sequences([text])
    pad_base = pad_sequences(seq_base, maxlen=MAX_LEN_BASE)
    pred_base_val = baseline.predict(pad_base, verbose=0)[0][0]
    
    label_base = "Positive" if pred_base_val > 0.5 else "Negative"
    conf_base = pred_base_val * 100 if label_base == "Positive" else (1 - pred_base_val) * 100

    # --- RAG+GRU PREDICTION ---
    # 1. Retrieve Context using TF-IDF
    query_vec = vectorizer.transform([text])
    similarities = query_vec * doc_vectors.T
    row = similarities.tocsr().getrow(0)
    
    if row.nnz > 0:
        top_k_indices = row.indices[np.argsort(row.data)[-2:]]
        contexts = [texts_train[idx] for idx in top_k_indices]
        compressed = " ".join([" ".join(c.split()[:20]) for c in contexts])
    else:
        compressed = ""
        
    # 2. Combine text with context
    final = text + " " + compressed

    # 3. Tokenize and Pad for RAG
    seq_rag = tokenizer.texts_to_sequences([final])
    pad_rag = pad_sequences(seq_rag, maxlen=MAX_LEN_RAG)

    # 4. Predict
    pred_rag_val = optimized.predict(pad_rag, verbose=0)[0][0]
    
    label_rag = "Positive" if pred_rag_val > 0.5 else "Negative"
    conf_rag = pred_rag_val * 100 if label_rag == "Positive" else (1 - pred_rag_val) * 100
    
    return label_base, conf_base, label_rag, conf_rag

acc_b = artifacts.get('acc_b', 0)
acc_o = artifacts.get('acc_o', 0)

print("\n==============================================")
print("🎬 IMDB RAG+GRU PREDICTION TERMINAL 🎬")
print("==============================================")
if acc_b > 0 and acc_o > 0:
    print(f"📊 Dataset Performance Stats:")
    print(f"   - Baseline GRU: {acc_b*100:.2f}%")
    print(f"   - Optimized RAG+GRU: {acc_o*100:.2f}%")
    print("==============================================")
print("Type 'quit' or 'exit' to stop.")

while True:
    sample = input("\nEnter a movie review to test: ")
    if sample.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break
        
    lb, cb, lr, cr = predict_both(sample)
    
    print(f"\n[Baseline GRU]      -> {lb} ({cb:.1f}% accuracy)")
    print(f"[Optimized RAG+GRU] -> {lr} ({cr:.1f}% accuracy)")
