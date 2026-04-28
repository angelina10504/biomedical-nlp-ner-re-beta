# VIVA PREPARATION GUIDE
## Biomedical Named Entity Recognition & Relation Extraction
### Angelina Gupta (220456)

---

# PART 1: PROJECT OVERVIEW

## What is this project about?

You built a complete biomedical NLP pipeline that does two things:

**Task 1 - Named Entity Recognition (NER):** Given a sentence from a medical paper, identify which words are disease names. For example, in "The patient was diagnosed with breast cancer", the system should tag "breast" as B-Disease and "cancer" as I-Disease.

**Task 2 - Relation Extraction (RE):** Given a sentence mentioning two drugs, classify what kind of interaction exists between them. For example, "Aspirin inhibits the metabolism of Warfarin" should be classified as a "mechanism" interaction.

## Why progressive methodology?

You didn't just build one model. You built a progression of increasingly sophisticated models for each task, so you could measure exactly what each technique contributes. This is a key design decision — it lets you make scientific claims like "adding the CRF layer improved F1 by +0.2054" rather than just saying "our model works well."

---

# PART 2: DATASETS — Know Your Data Cold

## NCBI Disease Dataset (for NER)

**Source:** HuggingFace parquet, originally from the National Center for Biotechnology Information.

**Size:** 5,433 training sentences, 924 validation sentences, 941 test sentences.

**Label format:** BIO tagging scheme with 3 labels:
- **O** = Outside (not a disease word) — ~95% of all tokens
- **B-Disease** = Beginning of a disease entity
- **I-Disease** = Inside/continuation of a disease entity

**Example:**
```
"hereditary breast cancer"
 B-Disease  I-Disease I-Disease
```

**Key statistics you should know:**
- Average sentence length: 25.29 tokens
- Max sentence length: 123 tokens
- 2,134 unique disease entities across the full dataset
- Top diseases: DM (176), APC (119), DMD (115), breast cancer (87)
- Most entities are 1-word, but many are multi-word (2-word, 3+ words)

**Viva question: "Why is the O class problem significant?"**
Answer: About 95% of tokens are O. If a model blindly predicts O for everything, it gets 95% token accuracy but identifies zero diseases. That's why you use entity-level F1 (seqeval) instead of token accuracy — it only counts a prediction as correct if the entire disease span matches exactly.

## DDI Corpus (for RE)

**Source:** DrugBank XML files from the DDICorpus GitHub repository.

**Size:** 26,005 training pairs, 5,265 test pairs.

**5 classes:**
- **no-relation** (85%) — drugs mentioned together but no interaction
- **effect** (6%) — clinical effect of combination (e.g., "increases blood pressure")
- **mechanism** (5%) — biochemical pathway (e.g., "inhibits CYP3A4 metabolism")
- **advise** (3%) — recommendation/warning (e.g., "avoid concurrent use")
- **int** (<1%) — general/unspecified interaction

**How was the data parsed?** (Notebook 01)
Your `parse_ddi()` function reads XML files, extracts `<sentence>` elements, finds all `<entity>` elements (drug names) and `<pair>` elements (drug pairs with their relation type). For each pair, it records: sentence text, drug1 name, drug2 name, relation label.

**Viva question: "Why is `int` class so hard?"**
Answer: Only 178 training examples (< 1% of data). The label means a generic/unspecified interaction with no strong lexical cue — unlike "mechanism" (which often has words like "inhibits", "metabolism") or "advise" (which has "avoid", "contraindicated"). Even with class weighting, models can't learn a reliable pattern from so few examples. Average F1 across all models is only 0.20.

---

# PART 3: WORD2VEC EMBEDDINGS (Notebook 04)

## What is Word2Vec?

Word2Vec (Mikolov et al., 2013) maps every word in a vocabulary to a dense vector of real numbers (in your case, 200 dimensions). The core idea is the **distributional hypothesis**: words appearing in similar contexts have similar meanings.

## The Math Behind Skip-gram

You used **Skip-gram** (sg=1), which works like this:

Given a center word w_c, predict its context words w_o within a window of size 5.

The objective is to maximize:

```
J = (1/T) * Σ_{t=1}^{T} Σ_{-5 ≤ j ≤ 5, j≠0} log P(w_{t+j} | w_t)
```

Where the probability is computed using softmax:

```
P(w_o | w_c) = exp(v'_{w_o} · v_{w_c}) / Σ_{w=1}^{V} exp(v'_w · v_{w_c})
```

In practice, this full softmax is too expensive (V is huge), so Word2Vec uses **negative sampling**: for each positive (center, context) pair, it randomly samples ~5 "negative" words and trains the model to distinguish real context words from random ones.

## Why domain-specific embeddings?

Generic Word2Vec trained on Wikipedia would put "apple" near "fruit" and "microsoft." But in your biomedical corpus:
- "tumor" should be near "malignancy", "carcinoma"
- "aspirin" should be near "ibuprofen" (both NSAIDs)
- Abbreviations like "DM", "COPD" are meaningless in generic corpora

## Your training setup

```python
Word2Vec(
    sentences   = tokenized_sentences,  # NCBI + DDI combined (~12k sentences)
    vector_size = 200,                  # each word → 200-dim vector
    window      = 5,                    # context window = 5 words each side
    min_count   = 2,                    # ignore words appearing < 2 times
    sg          = 1,                    # skip-gram (better for rare words)
    epochs      = 10,
    seed        = 42
)
```

**Vocabulary:** 8,345 words (min_count ≥ 2)

## The embedding matrix

For the downstream neural models, you built a lookup table:
- Shape: (8,387 × 200) — 8,387 words, each a 200-dim vector
- Index 0 = `<PAD>` (all zeros — padding token)
- Index 1 = `<UNK>` (random vector — unknown words)
- Words found in Word2Vec → copy their trained vector
- Words NOT found → random initialization from Uniform(-0.25, 0.25)

**Coverage:** The percentage of NER vocabulary words that had trained Word2Vec vectors (the rest got random vectors).

**Viva question: "Why skip-gram over CBOW?"**
Answer: Skip-gram is better for rare words and smaller datasets. In biomedical text, there are many rare disease names and drug names — skip-gram handles these better because it treats each (center, context) pair as a separate training example, giving rare words more gradient updates proportional to their context diversity.

---

# PART 4: NER MODELS — Deep Dive

## Model 1: Rule-Based NER (Notebook 02) — F1 = 0.5771

### How it works

**Step 1 — Build a disease dictionary:** Extract every annotated disease entity from the training set by concatenating B-Disease and I-Disease token sequences. Store both original-case and lowercased versions. Result: 2,233 entries.

**Step 2 — Medical suffix list:** 8 suffixes that indicate disease words:
- `-itis` (inflammation: hepatitis, arthritis)
- `-osis` (condition: fibrosis, neurosis)
- `-emia` (blood condition: leukemia, anemia)
- `-oma` (tumor: carcinoma, melanoma)
- `-pathy` (disease: neuropathy, myopathy)
- `-plasia` (growth abnormality: hyperplasia)
- `-trophy` (wasting: atrophy, dystrophy)
- `-ectomy` (surgical removal: appendectomy)

**Step 3 — Prediction algorithm:**
For each position in the sentence, try a **longest-match** dictionary scan (4-word window → 3 → 2 → 1). If a match is found, tag it as B-Disease + I-Disease. If no dictionary match, check if the single token has a medical suffix — if yes, tag as B-Disease.

### Results
- Precision: 0.5378 (many false positives from suffix rule)
- Recall: 0.6225 (misses unseen diseases)
- F1: 0.5771

### Why it fails
1. **OOV (Out-of-Vocabulary):** Any disease not in training data is invisible
2. **No context:** Can't distinguish "MS" (multiple sclerosis) from "MS" (manuscript)
3. **Partial matches:** Longest-match can fragment multi-word entities
4. **Suffix false positives:** "chromosomal instability" — suffix fires on non-disease words

---

## Model 2: CRF (Notebook 03) — F1 = 0.7825 (BEST NER MODEL)

### What is a Conditional Random Field?

A CRF models the entire label sequence jointly instead of making independent per-token decisions. The probability of a tag sequence y given input x is:

```
P(y | x) = (1/Z(x)) * exp(Σ_i Σ_k λ_k * f_k(y_{i-1}, y_i, x, i))
```

Where:
- f_k are **feature functions** (your hand-crafted features)
- λ_k are **learned weights** (how important each feature is)
- Z(x) is the **partition function** (normalizing constant so probabilities sum to 1)
- The sum is over all positions i and all feature functions k

### Two types of CRF features

**State features** = relationship between current word's properties and its tag:
- Example: `word.lower() = 'hepatitis' → B-Disease` (high weight)

**Transition features** = which tag follows which:
- `B-Disease → I-Disease` (high weight, this transition is common)
- `O → I-Disease` (large NEGATIVE weight, this transition is illegal in BIO)

### Your feature engineering (13+ features per token)

For the current word:
- `word.lower` — the word itself, lowercased
- `word.isupper` — is it ALL CAPS? (abbreviations like DM, COPD)
- `word.istitle` — is it Titlecase? (proper nouns)
- `word.isdigit` — is it a number?
- `word.len` — length of the word
- `word[-3:]` — last 3 characters (suffix)
- `word[-5:]` — last 5 characters (longer suffix)
- `word[:3]` — first 3 characters (prefix)
- `has_hyphen` — does it contain '-'? (common in medical terms)
- `has_digit` — does it contain any digit?
- `has_medical_suffix` — does it end in -itis, -oma, etc.?

For the previous word (-1) and next word (+1): word.lower, isupper, istitle, suffix, medical_suffix.

Also: `BOS` (beginning of sentence), `EOS` (end of sentence), `bigram` (current + next word).

### Hyperparameter tuning

Grid search over c1 (L1 regularization) and c2 (L2 regularization):
- c1 controls **sparsity** — pushes unimportant feature weights to exactly 0
- c2 controls **stability** — prevents any single weight from growing too large

Best hyperparameters: c1=0.1, c2=0.01 (validation F1 = 0.8092)

Final model trained on train+val combined, evaluated on test.

### Training algorithm

The CRF is trained using **L-BFGS** (Limited-memory Broyden-Fletcher-Goldfarb-Shanno), a quasi-Newton optimization method. At inference, it uses the **Viterbi algorithm** to find the globally optimal tag sequence.

### Results
- Precision: 0.8347 (very few false positives)
- Recall: 0.7365 (still misses some entities)
- F1: 0.7825

**Improvement over Rule-Based: +0.2054 (+35.6%)**

### Top learned features

The CRF learned that:
- `BOS` (beginning of sentence) strongly predicts B-Disease (weight +4.47) — diseases often start sentences in biomedical abstracts
- `word.isupper` predicts B-Disease (weight +3.86) — abbreviations like DM, APC
- Specific prefixes/suffixes like "obe" (obesity), "-mas" (melanomas) are strong indicators
- Previous word being "ovarian" or "breast" strongly predicts I-Disease for the next word

**Viva question: "Why is CRF better than rule-based?"**
Answer: Three reasons. First, it learns which feature combinations actually predict diseases from labeled data, rather than relying on fixed rules. Second, it uses context (previous and next word features). Third, it models label transitions — it learns that I-Disease should follow B-Disease, enforcing BIO validity.

---

## Model 3: BiLSTM (Notebook 05) — F1 = 0.6827

### What is an LSTM?

A plain RNN computes: `h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)`

Problem: **vanishing gradients** — as gradients flow backwards through many time steps, they shrink exponentially. The network forgets early context.

An LSTM fixes this with a **cell state** C_t (a "conveyor belt" for information) and three gates:

**Forget gate:** `f_t = σ(W_f * [h_{t-1}, x_t] + b_f)` — what fraction of old memory to erase

**Input gate:** `i_t = σ(W_i * [h_{t-1}, x_t] + b_i)` — what new information to write

**Output gate:** `o_t = σ(W_o * [h_{t-1}, x_t] + b_o)` — what part of memory to expose

**Cell update:**
```
C̃_t = tanh(W_c * [h_{t-1}, x_t] + b_c)     ← candidate new memory
C_t  = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t            ← mix old and new
h_t  = o_t ⊙ tanh(C_t)                        ← output
```

σ is sigmoid (output 0-1, acts like a valve). ⊙ is element-wise multiplication.

### Why Bidirectional?

A left-to-right LSTM only sees context before each token. But for NER, future context also matters:

> "T-cell **leukaemia** is rare" — knowing "is rare" after "leukaemia" confirms it's a disease.

A BiLSTM runs two LSTMs in parallel:
- Forward LSTM: reads left → right, produces h→_t
- Backward LSTM: reads right → left, produces h←_t

At each position t: `h_t = [h→_t ; h←_t]` (concatenation)

With hidden_size=256 and bidirectional=True: output = 256 × 2 = 512 dimensions per token.

### Your architecture

```
Input tokens (batch, 128)
    ↓ Embedding (8387 → 200, pre-trained Word2Vec)
    ↓ Dropout(0.5)
    ↓ BiLSTM (200 → 256×2 = 512, 2 layers, inter-layer dropout=0.3)
    ↓ Dropout(0.5)
    ↓ Linear (512 → 3)     ← one logit per class per token
    ↓ argmax per token      ← independent prediction
Output: 3 BIO tags per token
```

**Total trainable parameters:** ~4.2 million

### Training details

- **Loss:** CrossEntropyLoss(ignore_index=-100) — padding positions ignored
- **Optimizer:** Adam, lr=1e-3
- **Gradient clipping:** max_norm=5.0 (prevents exploding gradients)
- **ReduceLROnPlateau:** halves LR if val F1 stalls for 3 epochs
- **Early stopping:** patience=5 epochs
- **Batch size:** 32
- **Max sequence length:** 128 (truncate/pad)

### Results
- Precision: 0.6874
- Recall: 0.6781
- F1: 0.6827

### Why BiLSTM scores LOWER than CRF (important viva question!)

This is a **key finding** in your project. The BiLSTM (F1=0.6827) is worse than the CRF (F1=0.7825). Why?

1. **No structural decoding:** The BiLSTM applies argmax independently per token — it can predict illegal sequences like `O → I-Disease`. The CRF would never allow this.
2. **Dataset size:** With only ~5,000 training sentences, hand-crafted features (CRF) generalize better than what the LSTM can learn from scratch.
3. **Embedding coverage:** OOV tokens map to `<UNK>`, losing all form information that suffix features would capture.
4. **Illegal transitions:** The BiLSTM produced 41 illegal O → I-Disease transitions on the test set.

**This is a well-known phenomenon in low-resource biomedical NLP:** feature-rich linear models outperform neural models when labeled data is scarce and domain features are strong.

---

## Model 4: BiLSTM + CRF (Notebook 06) — F1 = 0.7514

### What does the CRF layer add?

The BiLSTM produces **emission scores** E (shape: seq_len × 3). The CRF adds a learned **transition matrix** T (shape: 3 × 3), where T[i,j] is the score for transitioning from tag i to tag j.

The score of a complete tag sequence y = (y_1, ..., y_T) is:

```
score(y) = Σ_{t=1}^{T} E[t, y_t] + Σ_{t=2}^{T} T[y_{t-1}, y_t]
```

**Training:** Maximize the log-probability of the correct sequence relative to ALL possible sequences. The denominator (partition function) is computed efficiently using the **forward algorithm** in O(T × K²) time.

**Inference:** The **Viterbi algorithm** finds the globally highest-scoring sequence in O(T × K²) time, guaranteeing structurally valid BIO output.

### Architecture

```
Input tokens
    ↓ Embedding (200d, Word2Vec)
    ↓ BiLSTM (2 layers, 256×2 = 512d, dropout=0.3)
    ↓ Dropout(0.5)
    ↓ Linear (512 → 3)     ← emission scores
    ↓ CRF (3×3 transitions) ← Viterbi decoding
Output: globally optimal tag sequence
```

**Trainable parameters:** 4,193,898 (almost identical to BiLSTM alone — the CRF only adds 3×3 = 9 transition parameters)

### Key improvement: illegal transition elimination
- BiLSTM alone: 41 illegal O → I-Disease transitions
- BiLSTM + CRF: only 6 illegal transitions (these are borderline cases where the transition score was close to 0)

### Results
- Precision: 0.8031
- Recall: 0.7059
- F1: 0.7514

### Why it didn't beat the standalone CRF

The BiLSTM-CRF's emission scores (from the BiLSTM) are less reliable than the CRF's hand-crafted features on this small dataset. The CRF layer compensates partially by enforcing structural constraints, but the emission quality is the bottleneck. With a larger training set, character-level embeddings, or a BERT encoder, the emission quality would surpass the hand-crafted ceiling.

---

## NER Error Analysis (Notebook 07) — Key Findings

### Error categories (for the CRF, your best model):
- **Multi-word misses:** 146 entities (e.g., "sporadic T-cell prolymphocytic leukaemia" — 6 tokens)
- **Abbreviation misses:** 63 entities (e.g., "T-PLL", "B-NHL")
- **Single-word misses:** 43 entities

### The non-monotone progression explained

```
Rule-Based (0.5771) → CRF (0.7825) → BiLSTM (0.6827) → BiLSTM+CRF (0.7514)
```

The CRF > BiLSTM result is not a bug — it's a real phenomenon. The CRF benefits from:
- Strong hand-crafted features (suffixes, prefixes, context window)
- Convex optimization (guaranteed convergence)
- Structural output via Viterbi

The BiLSTM must learn everything from scratch with limited data.

---

# PART 5: RELATION EXTRACTION MODELS — Deep Dive

## Model 1: Classical ML (Notebook 08) — Macro F1 = 0.5765 (BEST RE MODEL)

### Feature engineering (3 feature blocks, 8,019 total features)

**Block 1 — Between-entity TF-IDF (3,000 features):**
Extract the text between drug1 and drug2 in the sentence, then compute TF-IDF on bigrams. This captures interaction cues like "inhibits the metabolism of."

**What is TF-IDF?**
- TF (Term Frequency): how often a word appears in THIS sentence
- IDF (Inverse Document Frequency): log(total_documents / documents_containing_word) — down-weights common words like "the", "is"
- TF-IDF = TF × IDF — words frequent in one sentence but rare overall get high scores

**Block 2 — Full sentence TF-IDF (5,000 features):**
TF-IDF on the full sentence for overall context.

**Block 3 — Numeric/keyword features (19 features):**
- word_count_between (distance between drugs)
- sentence_length
- entity_order (which drug appears first)
- 16 keyword flags: binary indicators for words like "inhibit", "increase", "decrease", "avoid", "contraindicated", "potentiate", "toxicity", etc.

### Three classifiers trained

| Classifier | Macro F1 | Weighted F1 |
|---|---|---|
| **Logistic Regression** | **0.5765** | 0.8233 |
| Linear SVM | 0.5231 | 0.8196 |
| Random Forest | 0.4759 | 0.8431 |

All used `class_weight='balanced'` to handle imbalance.

**Why Logistic Regression wins:**
- High-dimensional sparse TF-IDF data favors linear models
- liblinear solver is designed for exactly this type of problem
- Regularization (C=1.0) prevents overfitting

**Why NOT RBF-SVM?** It computes distances between every pair of training points — on 20,000+ TF-IDF features this takes hours. LinearSVM is just as accurate on text data.

### Per-class F1 (Logistic Regression)
- advise: 0.6018
- effect: 0.5150
- int: 0.4387
- mechanism: 0.4390
- no-relation: 0.8877

---

## Model 2: BiLSTM + Self-Attention (Notebook 09) — Macro F1 = 0.3733

### Entity blinding

Before feeding sentences to the model, replace actual drug names with `DRUG1` and `DRUG2`. This forces the model to learn from context words (like "inhibits") rather than memorizing specific drug pairs. It also helps generalize to unseen drugs.

### The attention mechanism (key concept for viva)

A plain BiLSTM reads the whole sentence and produces one vector per timestep. The final hidden state squashes all information into one fixed-size vector. Drug-interaction sentences are often long (30-50 words), and the key evidence might be buried in the middle.

**Self-attention solves this by keeping ALL timestep outputs and learning which tokens matter most.**

The math:
```
score_t  = v · tanh(W · h_t)         ← scalar importance score for token t
alpha_t  = softmax(score_t)           ← normalize to sum to 1 (attention weights)
context  = Σ_t (alpha_t × h_t)       ← weighted sum of all hidden states
```

Where:
- W_attn is a learned matrix (256 → 128)
- v_attn is a learned vector (128 → 1)
- h_t is the BiLSTM output at position t (256-dim)

**Intuition:** For "DRUG1 inhibits the CYP3A4-mediated metabolism of DRUG2", attention learns to give high weights to "inhibits" and "metabolism" — the words that signal the interaction type.

### Architecture

```
Input (batch, 150)
    ↓ Embedding (8389 → 200, Word2Vec + DRUG1/DRUG2)
    ↓ BiLSTM (200 → 128×2 = 256, 1 layer)
    ↓ Self-Attention → (batch, 256) context vector
    ↓ Dropout(0.5) → Linear(256→128) → ReLU → Linear(128→5)
Output: 5-class logits
```

**Trainable parameters:** 2,082,286

### Class weight computation

```
raw_weight_i = total_samples / (num_classes × count_i)
weight_i = sqrt(raw_weight_i)     ← square-root smoothing
```

Square-root smoothing prevents extreme imbalances from dominating. Without it, the `int` class (178 examples) would get a weight 100× larger than `no-relation`, destabilizing training.

### Results
- Macro F1: 0.3733
- Weighted F1: 0.6979
- int class F1: 0.0606 (nearly zero)

---

## Model 3: TextCNN (Notebook 10) — Macro F1 = 0.3992

### How 1D convolution works on text

Your sentence is a matrix of shape (seq_len × 200) — each row is a word vector. A kernel of size k looks at k consecutive word vectors simultaneously.

Three parallel convolution branches:
- **kernel_size=2:** captures bigrams ("inhibits the", "of DRUG2")
- **kernel_size=3:** captures trigrams ("inhibits the metabolism")
- **kernel_size=4:** captures 4-grams ("avoid concurrent use of")

Each branch has 100 filters, so each produces 100 features.

**AdaptiveMaxPool1d(1):** After convolution, feature maps have different lengths (149, 148, 147). Max pooling collapses any length to exactly 1 value by picking the maximum — the single most informative n-gram.

### Architecture

```
Input (batch, 150)
    ↓ Embedding (8389 → 200)
    ↓ permute to (batch, 200, 150)     ← Conv1d expects (batch, channels, length)
    ↓ Three parallel branches:
       conv2(k=2) → ReLU → MaxPool → (batch, 100)
       conv3(k=3) → ReLU → MaxPool → (batch, 100)
       conv4(k=4) → ReLU → MaxPool → (batch, 100)
    ↓ Concatenate → (batch, 300)
    ↓ Dropout(0.5) → Linear(300→128) → ReLU → Linear(128→5)
Output: 5-class logits
```

### CNN vs BiLSTM trade-offs

| | TextCNN | BiLSTM |
|-|---------|--------|
| Speed | Fast (parallel) | Slower (sequential) |
| What it captures | Local n-gram patterns | Long-range dependencies |
| Good for | Short interaction phrases | Context spanning full sentence |

---

## RE Error Analysis (Notebook 11) — Key Findings

### Why Classical ML beats neural models (IMPORTANT viva question)

This is a **surprising and important result.** The simpler Logistic Regression (macro F1 = 0.5765) beats both neural models (BiLSTM+Attention: 0.3733, TextCNN: 0.3992).

**Four reasons:**

1. **Strong lexical signals:** Words like "inhibit", "avoid", "increase" directly signal interaction type. TF-IDF captures these explicitly. Neural models must learn these associations from scratch.

2. **Dataset size:** 26,000 examples with minority classes having < 1,000 examples each — not enough for neural networks to generalize well.

3. **Between-entity text:** The TF-IDF feature on text between the two drugs is a very powerful engineered feature that directly captures the interaction cue. The neural models must learn to focus on this region automatically.

4. **Class imbalance hurts neural models more:** Even with class weighting, gradients from the majority class dominate early training, causing the model to collapse toward predicting "no-relation."

**When would neural models win?**
- Larger datasets (100k+ examples)
- Pre-trained language models (BioBERT, PubMedBERT)
- Paraphrased or indirect interaction descriptions where keyword matching fails

### Hardest class: "int" (avg F1 = 0.20 across all models)

Only 178 training examples. The label means a generic/unspecified interaction with no strong lexical cue.

### Sentence length effect
- Accuracy highest for very short (1-15 tokens) and very long (76+) sentences
- Accuracy lowest for medium-length (36-50 tokens) — these sentences have complex structure where the interaction cue is far from the drug mentions

---

# PART 6: EVALUATION METRICS — Explain These Clearly

## Precision, Recall, F1 (for NER)

**Precision** = (correctly predicted entities) / (total predicted entities)
- "Of everything I predicted as a disease, how many actually are?"
- High precision → few false positives

**Recall** = (correctly predicted entities) / (total actual entities)
- "Of all actual diseases, how many did I find?"
- High recall → few false negatives

**F1** = 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean — punishes models that sacrifice one for the other
- A model with P=0.99, R=0.01 gets F1=0.02 (terrible, not 0.50)

**Why seqeval?** Entity-level evaluation — a prediction is only correct if the ENTIRE span matches (correct B-tag position AND all following I-tags). Token-level accuracy would be misleading due to the O-class dominance.

## Macro F1 vs Weighted F1 (for RE)

**Macro F1** = average F1 across all classes (each class counts equally)
- Treats the rare `int` class as equally important as `no-relation`
- Better for evaluating performance on minority classes

**Weighted F1** = weighted average by class frequency
- Dominated by `no-relation` (85% of data)
- Can be misleadingly high even if minority classes have F1 ≈ 0

**Your design choice:** You report macro F1 as the primary RE metric because the task's real value is identifying actual interactions, not just saying "no relation."

---

# PART 7: LIBRARIES — What Each One Does

| Library | What it does in your project |
|---|---|
| **PyTorch** | Deep learning framework — BiLSTM, BiLSTM-CRF, BiLSTM+Attention, TextCNN |
| **torch.nn** | Neural network layers (Embedding, LSTM, Linear, Conv1d, Dropout) |
| **torch.optim** | Optimizers (Adam) and learning rate schedulers (ReduceLROnPlateau) |
| **torchcrf** | CRF layer implementation (pytorch-crf package) |
| **scikit-learn** | Classical ML (LogisticRegression, LinearSVC, RandomForestClassifier, TF-IDF, LabelEncoder, train_test_split) |
| **sklearn-crfsuite** | CRF implementation for sequence labeling |
| **gensim** | Word2Vec training and KeyedVectors loading |
| **seqeval** | Entity-level NER evaluation (precision, recall, F1) |
| **pandas** | Data loading and manipulation (CSV, parquet) |
| **numpy** | Numerical operations, embedding matrices |
| **matplotlib/seaborn** | Visualization (bar charts, loss curves, confusion matrices, heatmaps) |
| **xml.etree.ElementTree** | Parsing DDI Corpus XML files |

---

# PART 8: KEY DESIGN DECISIONS — Why You Made Each Choice

### 1. BIO tagging (not IO or BIOES)
BIO distinguishes the first token of an entity (B-) from continuation tokens (I-). This handles adjacent entities: "cancer" [B] followed by "tumor" [B] = two separate entities, not one.

### 2. max_length = 128 for NER, 150 for RE
Average sentence length is ~25 tokens (NER) and ~30 tokens (RE). 128/150 covers 99%+ of sentences while keeping tensors manageable.

### 3. Embedding dimension = 200
A balance between expressiveness and training efficiency. 100 would be too compressed for biomedical vocabulary; 300 would require more data to train well.

### 4. hidden_size = 256 (NER BiLSTM) vs 128 (RE BiLSTM)
NER needs richer per-token representations (each token gets a tag). RE only needs one sentence-level vector, so smaller hidden size suffices.

### 5. Dropout = 0.5
Standard regularization rate for NLP. Randomly zeros out 50% of neurons during training, preventing co-adaptation and overfitting.

### 6. Adam optimizer (not SGD)
Adam adapts learning rates per-parameter using first and second moment estimates. Works well out-of-the-box for NLP without extensive LR tuning.

### 7. Gradient clipping (max_norm=5.0)
Prevents exploding gradients in RNNs. Without it, a single large gradient can destroy learned weights.

### 8. Entity blinding for RE (DRUG1/DRUG2 replacement)
Forces the model to learn from context words, not memorize specific drug pairs. Improves generalization to unseen drugs.

### 9. Square-root smoothed class weights for RE
Raw inverse-frequency weights would make the `int` class (178 examples) dominate training. Square-root dampens the effect while still giving minority classes more penalty.

---

# PART 9: ALL RESULTS SUMMARY

## NER Results (NCBI Disease corpus)

| Model | Precision | Recall | F1 | Improvement from baseline |
|---|---|---|---|---|
| Rule-Based | 0.5378 | 0.6225 | 0.5771 | — (baseline) |
| **CRF** | **0.8347** | **0.7365** | **0.7825** | **+0.2054 (+35.6%)** |
| BiLSTM | 0.6874 | 0.6781 | 0.6827 | +0.1056 |
| BiLSTM + CRF | 0.8031 | 0.7059 | 0.7514 | +0.1743 |

## RE Results (DDI corpus)

| Model | Macro F1 | Weighted F1 |
|---|---|---|
| **Classical ML (LogReg)** | **0.5765** | **0.8233** |
| BiLSTM + Attention | 0.3733 | 0.6979 |
| TextCNN (k=2,3,4) | 0.3992 | 0.7086 |

---

# PART 10: POTENTIAL VIVA QUESTIONS & ANSWERS

### Q1: "What would you do differently if you had more time?"
A: Three things. First, use a pre-trained language model like BioBERT or PubMedBERT — these have been pre-trained on billions of biomedical words and typically achieve F1 > 0.85 on NER and macro F1 > 0.80 on DDI RE. Second, add character-level CNNs to the BiLSTM-CRF (Lample et al., 2016) to capture morphological cues like suffixes without hand-crafting them. Third, try data augmentation for the rare RE classes (especially "int") — synonym replacement, back-translation, or entity swapping.

### Q2: "Why did you train your own Word2Vec instead of using pre-trained biomedical embeddings?"
A: Two reasons. First, it let me understand the embedding training process end-to-end as a learning exercise. Second, domain-specific embeddings capture the vocabulary and semantics of my exact datasets (NCBI disease names, DDI drug names). However, in a production setting, I would use pre-trained embeddings like BioWordVec (trained on PubMed) which have much larger vocabulary coverage.

### Q3: "Explain the Viterbi algorithm."
A: Viterbi is a dynamic programming algorithm that finds the highest-scoring tag sequence in O(T × K²) time, where T is sequence length and K is number of tags. Instead of checking all K^T possible sequences (exponential), it builds a table: at each position t, for each possible tag k, it stores the best score achievable ending with tag k at position t. It then backtracks from the highest-scoring final tag to recover the full sequence. This is what makes the CRF layer efficient — it explores all possible sequences implicitly without enumerating them.

### Q4: "What is the partition function Z(x) in the CRF?"
A: Z(x) is the sum of exp(score) over ALL possible tag sequences. It normalizes the scores into valid probabilities. Computing it naively would require summing over K^T sequences (exponential). The forward algorithm computes it in O(T × K²) by exploiting the chain structure — at each position, it only needs the accumulated scores from the previous position.

### Q5: "Why is your BiLSTM-CRF worse than the standalone CRF?"
A: The bottleneck is emission quality. The standalone CRF uses rich hand-crafted features (15+ per token) that encode strong biomedical knowledge (medical suffixes, word shape, context). The BiLSTM must learn these patterns from ~5,000 sentences — not enough to match hand-crafted features. The CRF layer on top helps by enforcing structural constraints (reducing illegal transitions from 41 to 6), but can't compensate for weaker emissions. With more data or pre-trained contextual embeddings (BERT), the BiLSTM emissions would surpass hand-crafted features.

### Q6: "What is the difference between attention and the CRF?"
A: They solve different problems. Attention is an **aggregation mechanism** — it produces a weighted average of all hidden states for sentence-level classification (RE). The CRF is a **structured output layer** — it models label transitions and finds the globally optimal tag sequence for token-level tagging (NER). Attention answers "which words matter most?" while CRF answers "what is the best label sequence?"

### Q7: "Why use macro F1 for RE instead of accuracy?"
A: Because 85% of pairs are "no-relation." A model that always predicts "no-relation" gets 85% accuracy but identifies zero actual interactions. Macro F1 averages F1 across all classes equally, so the rare but important classes (mechanism, effect, advise, int) count just as much as the dominant class.

### Q8: "What is entity blinding and why use it?"
A: Replacing actual drug names with generic tokens DRUG1 and DRUG2 before feeding sentences to the neural model. Without it, the model might memorize that "Aspirin + Warfarin = mechanism" from training data but fail on unseen drug pairs. With blinding, the model must learn from context words like "inhibits" and "metabolism," which generalize to any drug pair.

### Q9: "Explain max-pooling in TextCNN."
A: After a Conv1d filter slides across the sentence, it produces a feature map — one activation score per position. Max-pooling picks the single highest score across ALL positions. This is like asking: "Did this n-gram pattern appear ANYWHERE in the sentence?" It doesn't matter where "inhibits metabolism" appeared — max-pooling captures that it appeared at all. This also makes the output fixed-size regardless of sentence length.

### Q10: "What are the limitations of your approach?"
A: Five main limitations. (1) No pre-trained language model — BERT/BioBERT would significantly improve both tasks. (2) Word-level tokenization — subword tokenization (BPE) would handle OOV words better. (3) No character-level features for NER — would help with abbreviations and morphology. (4) Small training corpus for Word2Vec (~12k sentences vs. billions for public models). (5) No cross-task learning — the NER and RE modules are independent, but in practice, recognized entities could inform relation extraction.

---

# PART 11: KEY REFERENCES TO MENTION IN VIVA

- **Word2Vec:** Mikolov et al., 2013 — "Efficient Estimation of Word Representations in Vector Space"
- **CRF for NER:** Lafferty et al., 2001 — "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"
- **BiLSTM-CRF:** Lample et al., 2016 — "Neural Architectures for Named Entity Recognition" (state-of-the-art before BERT)
- **TextCNN:** Kim, 2014 — "Convolutional Neural Networks for Sentence Classification"
- **Attention:** Bahdanau et al., 2015 — "Neural Machine Translation by Jointly Learning to Align and Translate"
- **NCBI Disease corpus:** Doğan et al., 2014
- **DDI Corpus:** Herrero-Zazo et al., 2013
- **BioBERT:** Lee et al., 2020 — the model that would improve your results significantly
