# SentimentLens — Advanced CNN-BiLSTM-Attention Sentiment Analysis

## 1. The Architecture in Detail

This project implements a hybrid **CNN-BiLSTM-Attention** architecture in PyTorch. Let's break down exactly what happens and why this architecture was chosen.

### The Flow:
`Raw Text` → `Clean & Negation Fusion` → `Tokenize (Vocab: 35k)` → `Pad (Length: 200)` → `Embedding (128-dim)` → `Parallel CNNs (k=3, 5, 7)` → `Concatenate` → `2-Layer BiLSTM (64 units)` → `Self-Attention` → `Dense Head` → `Sigmoid Output (Binary Sentiment)`

### How do the individual components work? (The Brutal Breakdown)

*   **RNN (Recurrent Neural Network):** 
    *   **How it works:** RNNs are the grandfather of sequence models. They process text sequentially, token by token. At each step, they take the current word and the "hidden state" (memory) from the previous word to compute a new hidden state. 
    *   **Is it used here?** **NO.** Pure vanilla RNNs are completely obsolete and are *not* used in this architecture. Why? Because they suffer from the "vanishing gradient problem." If you feed an RNN a 200-word review, by the time it reaches the 200th word, the mathematical gradients used to update the weights shrink to near zero, meaning it completely "forgets" the context of the first 20 words. They are practically useless for long reviews.
*   **LSTM (Long Short-Term Memory):** 
    *   **How it works:** LSTMs are a specific, mathematically complex type of RNN built to fix the vanishing gradient problem. They introduce a "cell state" (a long-term memory highway) and three gates (Forget, Input, Output). 
        *   *Forget Gate:* Looks at the new word and decides what irrelevant past context to throw away.
        *   *Input Gate:* Decides what new information from the current word to add to the memory.
        *   *Output Gate:* Decides what part of the memory to output to the next layer.
    *   **The flaw:** They only read left-to-right. If the sentence is "The movie was not entirely terrible", when the LSTM reads "not", it doesn't yet know "terrible" is coming.
*   **BiLSTM (Bidirectional LSTM):** 
    *   **How it works:** This is what we actually use. It is literally two LSTMs stacked together. One reads the text left-to-right (from start to finish), and the other reads right-to-left (from finish to start). The outputs are then concatenated. 
    *   **The benefit:** When the model analyzes the word "not", it already has the context of "terrible" from the backward LSTM. It possesses total contextual awareness of both past and future words at every single timestep.
    *   **Code Map:** Instantiated at [`backend/model.py:L100-L107`](file:///d:/dl/dl_lab/backend/model.py#L100-L107). Forward pass at [`L145`](file:///d:/dl/dl_lab/backend/model.py#L145).
*   **CNN (Convolutional Neural Network):** 
    *   **How it works:** Typically used in images, 1D CNNs slide a "window" (kernel) over the text to extract local patterns (n-grams). 
*   **Self-Attention (The Game Changer):**
    *   **How it works:** In a standard BiLSTM, you usually just take the very last hidden state (at the end of the sentence) and feed it to the dense layer to guess the sentiment. This is a massive bottleneck—you are forcing the network to compress a 200-word review into a single vector. 
    *   Self-Attention fixes this. Instead of taking the last state, it looks at *all 200 outputs* from the BiLSTM simultaneously. It calculates an alignment score (using a smaller neural network, often called Bahdanau attention) to determine the "importance" of each word relative to the overall sentiment. 
    *   **How it improves performance:** If a review is 190 words of boring plot summary and 10 words of intense criticism ("However, the acting was an absolute catastrophe"), Self-Attention assigns massive mathematical weight to those 10 words and near-zero weight to the plot summary. The final vector passed to the output layer is a weighted sum of the entire sequence, bypassing the LSTM's memory bottleneck entirely.
    *   **Code Map:** Defined at [`backend/model.py:L27-L48`](file:///d:/dl/dl_lab/backend/model.py#L27-L48). Forward pass executed at [`L148`](file:///d:/dl/dl_lab/backend/model.py#L148).
### Structural Modifications & How It Stands Out

1.  **Parallel Conv1D with Sequence Preservation:** 
    Instead of standard pooling that destroys the temporal timeline, this model uses three parallel `Conv1D` layers with kernel sizes 3, 5, and 7, and uses `padding=kernel_size//2`. This extracts 3-gram, 5-gram, and 7-gram features *while keeping the sequence length exactly 200*. 
2.  **Concatenation over Pooling:** 
    The outputs of the three CNNs are concatenated along the feature dimension, meaning every word now has features representing its surrounding 3, 5, and 7 words.
3.  **BiLSTM on top of CNN Features:**
    The BiLSTM does not process raw word embeddings; it processes the rich, n-gram-aware features outputted by the CNN.
4.  **Negation Fusion (Preprocessing):**
    "not good" is fused into "not_good" before tokenization. This structurally prevents the model from splitting negations and missing their meaning.

### Why not just use LLMs or pure RNNs?

*   **Why not LLMs (GPT, Llama)?** LLMs are massive, require GPUs for inference, have high latency, and are computationally expensive. This CNN-BiLSTM has only ~4.6M parameters, runs lightning-fast on a CPU, and takes up ~19MB of disk space. For a strict classification task, it is vastly more efficient.
*   **Why not pure RNN/LSTM?** LSTMs are slow to train because they are strictly sequential. By putting CNNs *before* the LSTM, the CNNs rapidly extract local feature representations in parallel, making the LSTM's job easier and allowing it to focus purely on high-level long-range sentiment shifts rather than basic phrase recognition.

---

## 2. Training Process & Dataset

### The Dataset Construction (~300K rows)
To make the model robust, it is trained on a highly diverse, engineered dataset:
1.  **IMDB Reviews (50,000 rows):** Standard movie reviews. Good for long, narrative text.
2.  **Yelp Reviews (200,000 rows):** Sampled from a larger dataset. Good for slang, food, and product reviews.
3.  **Sarcasm Augmentation (~50,000 rows):** Sourced from News Headlines, SARC (Reddit), and ~170 hand-crafted examples. Sarcasm is mapped to binary (Sarcastic = Negative proxy, Non-Sarcastic = Positive proxy).

**Is this data enough?** 
Yes and no. For standard consumer reviews, 300K balanced rows are more than enough to achieve high accuracy. However, binary sentiment (Pos/Neg) is inherently limiting. Real human text has nuanced emotions (confusion, irony, relief).

### The Training Process
1.  **Preprocessing:** Lowercasing, HTML stripping, negation fusion, punctuation separation.
2.  **Tokenization:** A custom `SimpleTokenizer` builds a vocabulary of the top 35,000 words. (Increased from 20k to capture Yelp/Reddit slang).
3.  **Padding:** Sequences are padded/truncated to 200 tokens.
4.  **Training Setup:** 
    *   **Loss:** Binary Cross-Entropy (`BCELoss`)
    *   **Optimizer:** Adam (`1e-3` learning rate)
    *   **Scheduler:** ReduceLROnPlateau (halves LR if validation loss stalls).
    *   **Early Stopping:** Patience of 3 epochs.

---

## 3. Proof of Work & Expected Outcomes

### In-Distribution Performance (The Training Set)
On the 20% hold-out test set from the IMDB+Yelp+Sarcasm corpus, the model achieves:
*   **Test Accuracy:** ~88.9%
*   **F1-Score:** ~0.89

### Out-of-Distribution Performance (The Brutal Truth)
To truly prove if it works, a batch test was run on the **GoEmotions** dataset (Reddit comments mapped to 27 emotions). We mapped positive emotions (joy, love) to Positive and negative (anger, disgust) to Negative.
*   **GoEmotions Accuracy:** ~63.06%
*   **GoEmotions F1-Score:** 0.7217
*   **False Negatives (30k):** The model often classifies subtle positive emotions as negative.

**What does this mean?** 
The model is highly optimized for explicit reviews (movies, restaurants, products). It struggles with highly nuanced conversational text (like Reddit) where sentiment is implied or extremely short. The sarcasm injection helps, but mapping 27 complex emotions to a binary 0/1 is noisy. 

### Brutal Evaluation: Will this Architecture Actually Work?

**The Short Answer:** Yes, but only for the exact job it was designed to do. 

**What data can it handle flawlessly?**
*   **Explicit Consumer Reviews:** Movie reviews, Amazon product reviews, App store feedback, Yelp restaurant reviews. If a human is consciously writing an evaluation with clear sentiment-bearing adjectives ("great", "awful", "highly recommend", "waste of money"), this CNN-BiLSTM-Attention model will absolutely crush it, operating at near 90% accuracy. The CNN quickly flags the keyword n-grams, the BiLSTM resolves the context, and Attention pinpoints the thesis statement.
*   **Medium-to-Long Form Text:** It is padded to 200 tokens. It thrives when there is enough context to establish a baseline sentiment.

**What are its massive limitations?**
*   **Nuanced / Implicit Sentiment:** Look at the GoEmotions test (63% accuracy). If someone writes "I can't believe they did that," is that positive (amazement) or negative (anger)? The model struggles heavily with zero-context conversational text (like Reddit threads or short tweets).
*   **Sarcasm Failure Rate:** Despite injecting 50k sarcastic rows into the training data, sarcasm heavily relies on real-world knowledge that this model does not possess. If someone writes "Oh brilliant, another flat tire," the model sees "brilliant" (strong positive weight) and "flat tire" (neutral/mild negative) and might misclassify it as positive. It lacks the semantic reasoning of an LLM.
*   **Binary Restriction:** The real world isn't just 1 (Positive) or 0 (Negative). Human emotion includes neutrality, mixed feelings, fear, and curiosity. Forcing a continuous emotional spectrum into a binary sigmoid output guarantees failure on edge cases.

**Final Verdict:** 
Do not use this for generalized conversational AI or subtle intent detection. But if you need a blazing-fast, lightweight, cheap-to-run classifier to mass-process 100,000 product reviews per minute without paying OpenAI API fees, this architecture is a lethal, highly optimized weapon.

**Expected Outcomes in Production:**
If you feed it standard reviews, expect ~85-90% accuracy. If you feed it Twitter/X arguments, expect ~60-70% accuracy.

---

## 4. Tech Stack Used

*   **Deep Learning Framework:** PyTorch (CPU-optimized for production)
*   **Backend / API:** FastAPI, Uvicorn, Pydantic
*   **Data Processing:** Pandas, NumPy, Scikit-learn
*   **Frontend:** Vanilla HTML, CSS, JavaScript (Zero dependencies)
*   **Deployment Config:** Render (`render.yaml` provided)

## Quick Start

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the Model (~5–10 min on CPU/GPU)
```bash
cd backend
python train.py
```
*(Note: This downloads IMDB, Yelp, and Sarcasm datasets automatically. Ensure you have ~300MB of disk space).*

### 3. Start the Backend API
```bash
cd backend
uvicorn app:app --reload --port 8000
```

### 4. Start the Frontend
```bash
cd frontend
python -m http.server 5500
```
Open **http://localhost:5500** in your browser.
