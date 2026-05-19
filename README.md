# Biomedical NLP Project
### Disease Named Entity Recognition & Drug-Drug Interaction Relation Extraction

> **Group 22** — BTech CSE | Scientific Writing + NLP Course Project

---

## Overview

This project builds biomedical NLP systems **from scratch** (no pre-trained LLMs or BERT) for two tasks:

1. **Named Entity Recognition (NER)** — identify disease mentions in biomedical text using the NCBI Disease dataset
2. **Relation Extraction (RE)** — classify drug-drug interactions (DDI) from the DDI Corpus

Models are developed progressively, from simple rule-based baselines to deep learning architectures, to understand the contribution of each technique.

---

## Project Structure

```
biomedical-nlp-project-beta/
├── data/
│   ├── ncbi_train.csv          # NCBI Disease — train split
│   ├── ncbi_val.csv            # NCBI Disease — validation split
│   ├── ncbi_test.csv           # NCBI Disease — test split
│   ├── ddi_train.csv           # DDI Corpus — train split
│   └── ddi_test.csv            # DDI Corpus — test split
│
├── models/
│   ├── word2idx.json           # Vocabulary index (word → int)
│   ├── embedding_matrix.npy    # Pre-trained Word2Vec embeddings (200-dim)
│   ├── bilstm_ner.pt           # BiLSTM NER checkpoint
│   ├── bilstm_crf_ner.pt       # BiLSTM+CRF NER checkpoint
│   ├── bilstm_attention_re.pt  # BiLSTM+Attention RE checkpoint
│   └── cnn_re.pt               # TextCNN RE checkpoint
│
├── results/
│   ├── ner_results.json        # NER comparison table
│   ├── re_results.json         # RE comparison table
│   ├── 05_training_history.json
│   ├── 06_training_history.json
│   └── *.png                   # Generated figures
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_rule_based_ner.ipynb
    ├── 03_crf_ner.ipynb
    ├── 04_word2vec_embeddings.ipynb
    ├── 05_bilstm_ner.ipynb
    ├── 05_figures.ipynb        # ROC/AUC, Precision/Recall, Validation Curve for BiLSTM NER
    ├── 06_bilstm_crf_ner.ipynb
    ├── 06_figures.ipynb        # ROC/AUC, Precision/Recall, Validation Curve for BiLSTM+CRF NER
    ├── 07_ner_error_analysis.ipynb
    ├── 08_classical_ml_re.ipynb
    ├── 09_bilstm_attention_re.ipynb
    ├── 10_cnn_re.ipynb
    ├── 11_re_error_analysis.ipynb
    └── 12_final_results.ipynb
```

---

## Datasets

| Dataset | Task | Size |
|---------|------|------|
| [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) | NER | 793 train / 100 val / 100 test sentences |
| [DDI Corpus](https://hulat.inf.uc3m.es/DrugDDI/DrugDDI.html) | RE | 27,792 sentence pairs |

### NER Label Schema (BIO tagging)
| Label | ID | Meaning |
|-------|----|---------|
| `O` | 0 | Outside any entity |
| `B-Disease` | 1 | Beginning of a disease mention |
| `I-Disease` | 2 | Inside a disease mention |

### RE Label Schema (DDI)
| Label | Meaning |
|-------|---------|
| `NO-REL` | No drug-drug interaction |
| `effect` | Drug A affects the effect of Drug B |
| `mechanism` | Pharmacokinetic mechanism described |
| `advise` | Clinical recommendation given |
| `int` | Generic interaction mentioned |

---

## Models

### NER — Progressive Development

| Notebook | Method | Test F1 |
|----------|--------|---------|
| `02` | Rule-Based (regex + dictionary) | 0.5771 |
| `03` | CRF (sklearn-crfsuite) | 0.7825 |
| `05` | BiLSTM (Word2Vec embeddings) | — |
| `06` | BiLSTM + CRF | — |

### RE — Progressive Development

| Notebook | Method |
|----------|--------|
| `08` | Classical ML (Logistic Regression, SVM, Random Forest) |
| `09` | BiLSTM + Self-Attention |
| `10` | TextCNN (parallel conv branches, kernel sizes 2/3/4) |

---

## Architecture Details

### Word2Vec Embeddings (Notebook 04)
- Algorithm: Skip-gram
- Dimensions: 200
- Trained on: NCBI Disease + DDI corpus combined
- Saved as: `models/word2idx.json` + `models/embedding_matrix.npy`

### BiLSTM NER (Notebook 05)
- Embedding: 200-dim Word2Vec (fine-tuned)
- BiLSTM: hidden=256, 2 layers, bidirectional → 512-dim output
- Dropout: 0.3 (between LSTM layers), 0.5 (before FC)
- Output: Linear(512 → 3), token-level cross-entropy loss
- Sequence length: 128 tokens (padded/truncated)

### BiLSTM + CRF NER (Notebook 06)
- Same BiLSTM backbone as above
- Adds CRF layer (torchcrf) for globally optimal tag sequence decoding
- Training: CRF negative log-likelihood loss
- Decoding: Viterbi algorithm — enforces valid BIO transitions
- Key benefit: eliminates illegal `O → I-Disease` transitions

### BiLSTM + Attention RE (Notebook 09)
- Embedding: 200-dim Word2Vec
- BiLSTM encoder → self-attention pooling → classification head
- 5-class output (NO-REL, effect, mechanism, advise, int)

### TextCNN RE (Notebook 10)
- 3 parallel Conv1d branches (kernel sizes 2, 3, 4)
- 100 filters per branch → max-pool → concat (300-dim) → FC
- Dropout + ReLU activations

---

## Setup & Running

### Requirements

```bash
pip install torch numpy pandas scikit-learn gensim
pip install pytorch-crf seqeval matplotlib
```

### Running on Google Colab (Recommended)

All notebooks mount Google Drive automatically. Ensure the project folder is at:
```
MyDrive/biomedical-nlp-project-beta/
```

Then open any notebook in Colab and run **Runtime → Run All**.

### Running Locally (VS Code)

Replace the Drive mount cell at the top of each notebook with:
```python
import os
os.chdir('/path/to/biomedical-nlp-project-beta')
```

> **macOS note:** If you get a `KeyboardInterrupt` on sklearn import, add this as the first cell:
> ```python
> import os
> os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
> ```

### Notebook Run Order

```
04 → 02 → 03 → 05 → 06 → 05_figures → 06_figures
                  ↓
             08 → 09 → 10
```

> Always run notebook **04** first to generate Word2Vec embeddings and vocabulary files.

---

## Evaluation Figures

Each `*_figures.ipynb` notebook generates three plots (requires the trained model checkpoint and test CSV):

- **ROC / AUC curves** — one-vs-rest per class, using emission probabilities from the model
- **Precision & Recall** — per-class bar chart
- **Validation curve** — training loss, validation loss, and validation F1 over epochs (requires `results/0X_training_history.json` saved during training)

---

## Key Design Decisions

**Why no BERT / pre-trained LLMs?**
This project deliberately builds from scratch to understand each component's contribution: embeddings → sequence modeling → structured prediction.

**Why is CRF not used for RE?**
CRF is a sequence labeling technique (assigns a label to each token in a sequence). RE is a sentence-level classification task (assign one label to a sentence/pair), so CRF is not applicable.

**Why `_get_emissions()` for BiLSTM+CRF figures?**
`CRF.decode()` returns hard tag sequences (Viterbi), not probabilities. To compute ROC/AUC, we extract raw emission logits via `model._get_emissions()` and apply softmax separately.

---

## Results Summary

Results are saved incrementally to `results/ner_results.json` and `results/re_results.json` as each notebook runs, and compiled into a final comparison table in notebook `12_final_results.ipynb`.

---

## References

- Doğan, R. I., Leaman, R., & Lu, Z. (2014). NCBI disease corpus. *Journal of Biomedical Informatics*.
- Herrero-Zazo, M. et al. (2013). The DDI corpus. *Journal of Biomedical Informatics*.
- Lample, G. et al. (2016). Neural architectures for named entity recognition. *NAACL*.
- Kim, Y. (2014). Convolutional neural networks for sentence classification. *EMNLP*.
