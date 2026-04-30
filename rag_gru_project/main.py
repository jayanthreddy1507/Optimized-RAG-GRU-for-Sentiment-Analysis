import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import imdb

# ---------------------------
# 1. LOAD IMDB DATASET
# ---------------------------
print("[1/9] Loading IMDB dataset (50,000 reviews)...")
VOCAB_SIZE = 10000
MAX_LEN_BASE = 50
MAX_LEN_RAG = 200

(X_train_seq, y_train), (X_test_seq, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# ---------------------------
# 2. DECODE SEQUENCES FOR TF-IDF
# ---------------------------
print("[2/9] Decoding sequences to text for RAG retrieval...")
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

def decode_review(seq):
    return ' '.join([reverse_word_index.get(i, '?') for i in seq])

texts_train = [decode_review(seq) for seq in X_train_seq]
texts_test = [decode_review(seq) for seq in X_test_seq]

all_texts = texts_train + texts_test

# ---------------------------
# 3. FAST RETRIEVAL SYSTEM (TF-IDF + Sparse Math)
# ---------------------------
print("[3/9] Building TF-IDF matrix for context retrieval...")
vectorizer = TfidfVectorizer(max_features=10000)
# Fit on training data only to avoid data leakage
doc_vectors = vectorizer.fit_transform(texts_train)

def build_rag_dataset(texts, base_vectors, is_train=False):
    rag_texts = []
    # Transform queries
    query_vectors = vectorizer.transform(texts)
    
    # Fast sparse matrix multiplication: queries x base_docs^T
    # Result is a sparse matrix of shape (num_queries, num_base_docs)
    similarities = query_vectors * base_vectors.T
    
    # Convert to efficient format for row-wise operations
    similarities = similarities.tocsr()
    
    for i in range(len(texts)):
        row = similarities.getrow(i)
        
        # Get indices of top 2 non-zero elements
        if row.nnz > 0:
            # Sort the sparse row values and get the top indices
            # .indices contains the column indices (doc IDs)
            # .data contains the similarity scores
            top_k_indices = row.indices[np.argsort(row.data)[-2:]]
            
            # If it's training data, we might retrieve the document itself (similarity 1.0)
            # We filter it out to prevent trivial copying
            if is_train:
                top_k_indices = [idx for idx in top_k_indices if idx != i]
            
            # Get the text of the retrieved contexts
            contexts = [texts_train[idx] for idx in top_k_indices]
            
            # Compress context to first 20 words each to keep sequences manageable
            compressed = " ".join([" ".join(c.split()[:20]) for c in contexts])
        else:
            compressed = ""
            
        rag_texts.append(texts[i] + " " + compressed)
        
        if (i+1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(texts)} RAG queries")
            
    return rag_texts

print("[4/9] Generating RAG contexts for Training Data...")
rag_train_texts = build_rag_dataset(texts_train, doc_vectors, is_train=True)

print("[5/9] Generating RAG contexts for Test Data...")
rag_test_texts = build_rag_dataset(texts_test, doc_vectors, is_train=False)

# ---------------------------
# 4. TOKENIZATION & PADDING
# ---------------------------
print("[6/9] Tokenizing and Padding Sequences...")
# Use existing integer sequences for baseline, just pad them
X_train_base = pad_sequences(X_train_seq, maxlen=MAX_LEN_BASE)
X_test_base = pad_sequences(X_test_seq, maxlen=MAX_LEN_BASE)

# Tokenize RAG texts
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts_train) # Fit on training text

X_train_rag = pad_sequences(tokenizer.texts_to_sequences(rag_train_texts), maxlen=MAX_LEN_RAG)
X_test_rag = pad_sequences(tokenizer.texts_to_sequences(rag_test_texts), maxlen=MAX_LEN_RAG)

# ---------------------------
# 5. BASELINE GRU MODEL
# ---------------------------
def build_baseline():
    inp = Input(shape=(MAX_LEN_BASE,))
    x = Embedding(VOCAB_SIZE, 64)(inp)
    x = GRU(64)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# ---------------------------
# 6. OPTIMIZED GRU (Context Fusion)
# ---------------------------
def build_optimized():
    query_input = Input(shape=(MAX_LEN_RAG,))

    x = Embedding(VOCAB_SIZE, 64)(query_input)
    x = GRU(64, return_sequences=False)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = Model(query_input, out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# ---------------------------
# 7. TRAIN BASELINE
# ---------------------------
print("\n[7/9] Training Baseline GRU Model...")
baseline = build_baseline()
baseline.fit(X_train_base, y_train, batch_size=128, epochs=5, validation_data=(X_test_base, y_test))

loss_b, acc_b = baseline.evaluate(X_test_base, y_test, verbose=0)

# ---------------------------
# 8. TRAIN OPTIMIZED MODEL
# ---------------------------
print("\n[8/9] Training Optimized RAG+GRU Model...")
optimized = build_optimized()
optimized.fit(X_train_rag, y_train, batch_size=128, epochs=5, validation_data=(X_test_rag, y_test))

loss_o, acc_o = optimized.evaluate(X_test_rag, y_test, verbose=0)

# ---------------------------
# 9. RESULTS
# ---------------------------
print("\n[9/9] RESULTS COMPARISON")
print(f"Baseline GRU Accuracy: {acc_b*100:.2f}%")
print(f"Optimized RAG+GRU Accuracy: {acc_o*100:.2f}%")

# ---------------------------
# 10. TEST SAMPLE
# ---------------------------
def predict(text):
    query_vec = vectorizer.transform([text])
    similarities = query_vec * doc_vectors.T
    row = similarities.tocsr().getrow(0)
    
    if row.nnz > 0:
        top_k_indices = row.indices[np.argsort(row.data)[-2:]]
        contexts = [texts_train[idx] for idx in top_k_indices]
        compressed = " ".join([" ".join(c.split()[:20]) for c in contexts])
    else:
        compressed = ""
        
    final = text + " " + compressed

    seq = tokenizer.texts_to_sequences([final])
    pad = pad_sequences(seq, maxlen=MAX_LEN_RAG)

    pred = optimized.predict(pad)[0][0]
    return "Positive" if pred > 0.5 else "Negative"

print("\n[SAMPLE TEST]")
sample = "This movie is bored"
print("Input:", sample)
print("Prediction:", predict(sample))

# ---------------------------
# 11. SAVE MODELS AND ARTIFACTS
# ---------------------------
import pickle
print("\n[10/10] Saving models and artifacts for future use...")
baseline.save('baseline_model.keras')
optimized.save('optimized_model.keras')
with open('rag_artifacts.pkl', 'wb') as f:
    pickle.dump({
        'vectorizer': vectorizer,
        'tokenizer': tokenizer,
        'doc_vectors': doc_vectors,
        'texts_train': texts_train,
        'acc_b': acc_b,
        'acc_o': acc_o
    }, f)
print("Saved successfully! You can now use predict.py to test new inputs instantly.")