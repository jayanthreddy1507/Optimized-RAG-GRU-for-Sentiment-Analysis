# 🎬 Optimized RAG+GRU for Sentiment Analysis

A Retrieval-Augmented Generation approach for context-enhanced IMDB movie review classification using GRU networks.

---
PDF-RESEARCH PAPER--LINK [RAG-GRU paper.docx](https://github.com/user-attachments/files/27251106/RAG-GRU.paper.docx)


## 📌 Overview

This project implements and benchmarks two sentiment analysis architectures on the **IMDB Movie Reviews Dataset (50,000 reviews)**:

| Model | Accuracy | Sequence Length |
|-------|----------|-----------------|
| Baseline GRU | 85.31% | 50 tokens |
| **Optimized RAG+GRU** | **86.58%** | **200 tokens** |

The key innovation is **Retrieval-Augmented Generation (RAG)**: instead of classifying each review in isolation, the model dynamically retrieves the top-2 most semantically similar training reviews using TF-IDF and fuses their compressed content with the input before feeding it to the GRU.

---

## 🧠 How It Works

```
Raw Review Text
      │
      ▼
TF-IDF Vectorizer ──► Sparse Similarity Search (25,000 docs)
      │                              │
      │                Top-2 Similar Reviews Retrieved
      │                              │
      └──────── Context Fusion (review + 20 tokens each) ────►
                                     │
                              Tokenizer + Padding
                              (vocab=10,000, maxlen=200)
                                     │
                              Embedding Layer (dim=64)
                                     │
                              GRU (64 hidden units)
                                     │
                              Dense + Dropout (0.3)
                                     │
                              Sigmoid Output
                              (Positive / Negative)
```

---

## 📁 Project Structure

```
├── main.py                # Train both models and save artifacts
├── predict.py             # Interactive terminal for inference
├── baseline_model.keras   # Saved baseline GRU (generated after training)
├── optimized_model.keras  # Saved RAG+GRU model (generated after training)
├── rag_artifacts.pkl      # TF-IDF vectorizer, tokenizer, doc vectors
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### Install dependencies

```bash
pip install tensorflow numpy pandas scikit-learn
```

---

## 🚀 Usage

### Step 1 — Train both models

```bash
python main.py
```

This will:

1. Load and decode the IMDB dataset (50,000 reviews)
2. Build the TF-IDF retrieval index on 25,000 training documents
3. Generate RAG-enriched sequences for train and test sets
4. Train the **Baseline GRU** (5 epochs, batch=128)
5. Train the **Optimized RAG+GRU** (5 epochs, batch=128)
6. Print accuracy comparison and save all models + artifacts

> ⚠️ First run takes ~10–20 minutes due to RAG context generation for 50,000 reviews.

### Step 2 — Run interactive predictions

```bash
python predict.py
```

Both models load from saved files. Enter any movie review and get side-by-side predictions:

```
Enter a movie review to test: This film was an absolute masterpiece!

[Baseline GRU]      -> Positive (94.2% confidence)
[Optimized RAG+GRU] -> Positive (97.1% confidence)
```

---

## 🔬 Architecture Details

### Baseline GRU

```
Input(50,) → Embedding(10000, 64) → GRU(64) → Dense(32, relu) → Dense(1, sigmoid)
```

### Optimized RAG+GRU

```
Input(200,) → Embedding(10000, 64) → GRU(64) → Dense(64, relu) → Dropout(0.3) → Dense(1, sigmoid)
```

### Retrieval Mechanism

Given query **q**, the TF-IDF similarity to all training docs is computed as:

```
S = TF-IDF(q) × D^T     (sparse matrix multiplication)
Top-2 = argtop-K(S)
```

Retrieved contexts are truncated to 20 tokens each and concatenated:

```
enriched_input = q + " " + compress(doc_1) + " " + compress(doc_2)
```

---

## 📊 Results

```
[9/9] RESULTS COMPARISON
Baseline GRU Accuracy:      85.31%
Optimized RAG+GRU Accuracy: 86.58%
```

The **+1.27% improvement** demonstrates that contextual retrieval consistently improves classification without modifying the underlying GRU architecture — making RAG a modular, plug-in enhancement for any sequential classifier.

---

## 💡 Key Design Decisions

| Decision | Reason |
|----------|--------|
| TF-IDF over dense retrieval (FAISS) | CPU-friendly, sub-second sparse matmul on 25K docs |
| Top-K=2 contexts | Balance between context richness and sequence length |
| 20 tokens per context | Preserves topical signal while keeping maxlen=200 manageable |
| Dropout=0.3 in RAG model | Prevents overfitting to retrieved context patterns |
| Fit vectorizer on train only | Strict no-data-leakage policy |

---

## 📄 Report

A full academic report covering the mathematical formulation, system architecture diagrams, GRU gate equations, and benchmarking analysis is included in the repository as `RAG_GRU_Sentiment_Report.docx`.

---

## 👨‍💻 Authors

**Department of Computer Science Engineering**  
Hyderabad, Telangana, India — April 2026

---

## 📜 License

This project is for academic purposes.
