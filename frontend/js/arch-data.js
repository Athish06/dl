/**
 * dl Neural Architecture Technical Directory
 * In-depth technical specifications, training methods, tensor flows,
 * traversal examples, layman intuitions, and ablation studies for every layer in dl.
 * Strictly 0 emojis.
 */

window.ARCH_COMPONENTS = {
    bpe_tokenizer: {
        id: "bpe_tokenizer",
        title: "Byte Pair Encoding (BPE) Subword Tokenizer",
        shortName: "BPE Tokenizer",
        category: "Preprocessing & Token Representation",
        models: ["Model V2 (Multi-Task)", "Model V3 (Flagship)"],
        parameters: "30,000 Vocabulary Units (0 Trainable Weights)",
        inputShape: "Raw Text String (UTF-8 encoded)",
        outputShape: "Tensor (batch_size, 200) Int64 IDs + (batch_size, 200) Binary Mask",
        badge: "Rust-Engineered BPE",
        summary: "Data-driven subword segmentation algorithm that balances character-level flexibility with word-level semantic density, eliminating Out-Of-Vocabulary (OOV) tokens.",
        beginnerIntuition: "Think of BPE like Lego bricks. Instead of manufacturing a unique, pre-molded plastic toy for every single vehicle in the universe, Lego gives you a set of universal bricks. When you encounter a brand-new concept like 'unprompted', you don't fail—you simply assemble it from familiar bricks: 'un' + 'prompt' + 'ed'.",
        whatIsIt: `Byte Pair Encoding (BPE) is a hybrid tokenization algorithm originally developed for data compression and adapted for modern deep learning NLP by Sennrich et al. Unlike traditional word-level tokenizers that break text strictly on whitespace and punctuation, BPE treats text as a sequence of subword morphemes. Common words (like "the" or "failure") remain intact as single tokens, while rare, compound, or slang words (like "unprompted" or "cinematic") are broken into frequent sub-lexical building blocks ("un" + "prompted", "cinema" + "tic"). This ensures 100% vocabulary coverage with zero out-of-vocabulary (OOV) tokens.`,
        howIsItTrained: `The tokenizer is trained in Rust using the Hugging Face Tokenizers library via the following iterative optimization loop:
1. Base Initialization: The initial vocabulary is populated with all unique UTF-8 bytes and single unicode characters present across the corpus (256 byte tokens).
2. Frequency Counting: The algorithm scans all 230,000+ training sentences and counts the co-occurrence frequency of every adjacent symbol pair (e.g., 't' + 'h', 'i' + 'n', 'e' + 'r').
3. Greedy Merge: The single most frequent pair is merged into a new composite token (e.g., 't' + 'h' -> 'th').
4. Vocabulary Expansion: The merged token is added to the vocabulary, and all instances in the corpus are replaced with the new unit.
5. Iteration: Steps 2-4 are repeated exactly 30,000 times until reaching the target vocabulary capacity (|V| = 30,000).
6. Merge Table Export: The final merge sequence is serialized into tokenizer_v3.json for deterministic, ultra-fast O(N) tokenization at inference time.`,
        dataFed: `The BPE tokenizer was trained on the combined Tier-2 Hybrid training corpus (230,689 sentences):
- Stanford IMDB (50,000 long-form movie reviews)
- Rotten Tomatoes SST-2 (67,769 single-sentence movie critiques)
- SARC 2.0 Reddit Sarcasm Corpus (90,000 conversational forum comments)
- News Headlines Sarcasm Dataset (28,619 sarcastic and literal news headlines)
This diverse mixture guarantees that the subword vocabulary captures formal cinematic terminology, informal internet slang, contractions, and emphatic punctuation patterns.`,
        architectureRole: `In dl V2 and V3, the BPE tokenizer serves as the primary gateway between raw human text and neural tensor space. It converts arbitrary text sequences into normalized integer token tensors (batch_size, 200) accompanied by a binary attention padding mask. By decomposing informal expressions and rare vocabulary into shared morphemic units, it allows downstream GloVe embeddings and parallel CNNs to extract rich feature vectors even for words the model has never explicitly encountered during training.`,
        ablationStudy: `What happens if BPE Tokenization is absent?
- Out-Of-Vocabulary Catastrophe: In Model V1 (which uses whitespace word tokenization with a 35K cutoff), any word not in the dictionary is replaced by [UNK] (Unknown Token). On conversational Reddit data, V1 produces an average of 4.2 [UNK] tokens per review, blinding the model to critical keywords.
- Sarcasm Idiom Loss: Sarcastic exaggerations like "masterclass", "groundbreakingly", or "sooooo" are split cleanly by BPE into meaningful roots ("master" + "class", "so" + "##oo"), whereas whitespace tokenizers discard them as unknown.
- Empirical Impact: In our ablation benchmarks, replacing BPE with whitespace tokenization causes a 4.8% drop in overall sentiment accuracy and an 8.2% drop in sarcasm recognition on conversational text.`,
        inputOutputSpec: {
            inputType: "String (Python str / JavaScript string)",
            inputDimensions: "Arbitrary length character sequence",
            outputType: "torch.LongTensor (Int64)",
            outputDimensions: "input_ids: (batch_size, 200), attention_mask: (batch_size, 200)",
            specialTokens: "<pad> = 0, <unk> = 1, <s> = 2, </s> = 3"
        },
        traversalExample: {
            sampleText: "What an absolute masterpiece of cinematic failure. Brilliant writing!",
            steps: [
                {
                    stage: "Raw Input String Normalization",
                    concept: "What is Normalization? Raw human text often contains invisible byte quirks, inconsistent spacing, or trailing line breaks. Normalization cleans the sentence so that every downstream layer receives consistent, standardized text.",
                    simpleInput: "Raw text string: 'What an absolute masterpiece of cinematic failure. Brilliant writing!'",
                    simpleProcessing: "Standardizes unicode formatting (NFC), strips non-printable byte sequences, and trims boundary spaces.",
                    simpleOutput: "A clean 68-character UTF-8 string with punctuation and case preserved.",
                    impact: "Ensures identical human queries always produce identical token streams with zero parsing bugs.",
                    math: {
                        formula: "T_norm = Normalize_NFC(Strip_ZeroWidth(T_raw))",
                        inputTensor: "Raw byte stream (68 bytes)",
                        operation: "Unicode Normalization Form C (NFC) + whitespace trimming",
                        outputTensor: "Normalized string (68 characters, 68 UTF-8 bytes, 0 invalid byte encodings)"
                    },
                    input: "What an absolute masterpiece of cinematic failure. Brilliant writing!",
                    operation: "Normalize unicode (NFC), strip non-printable byte sequences, trim trailing whitespace, and preserve syntactic punctuation.",
                    output: "Normalized UTF-8 String (68 characters, 68 bytes, 0 invalid bytes):\n'What an absolute masterpiece of cinematic failure. Brilliant writing!'"
                },
                {
                    stage: "Pre-Tokenization Regex Split",
                    concept: "What is Pre-Tokenization? Before the AI can split words into subwords, it must separate words from punctuation marks so that exclamation marks and periods become independent clues.",
                    simpleInput: "The clean 68-character string.",
                    simpleProcessing: "Uses a regular expression rule to separate words and punctuation marks at linguistic boundaries.",
                    simpleOutput: "11 word and punctuation pieces: ['What', 'an', 'absolute', 'masterpiece', 'of', 'cinematic', 'failure', '.', 'Brilliant', 'writing', '!'].",
                    impact: "Isolates exclamation marks ('!') and periods ('.') so they can serve as standalone sarcasm indicators.",
                    math: {
                        formula: "Tokens = RegexSplit(r'\\w+|[^\\w\\s]+', T_norm)",
                        inputTensor: "Normalized string (len=68)",
                        operation: "Byte-level regular expression boundary splitting",
                        outputTensor: "11 isolated lexical fragments (Array length: 11)"
                    },
                    input: "Normalized string",
                    operation: "Byte-level regex rule (\\w+|[^\\w\\s]+) splitting string into linguistic candidate words and punctuation boundaries.",
                    output: "11 isolated lexical fragments (Array length: 11):\n['What', 'an', 'absolute', 'masterpiece', 'of', 'cinematic', 'failure', '.', 'Brilliant', 'writing', '!']"
                },
                {
                    stage: "Subword Decomposition (30,000 BPE Merge Rules)",
                    concept: "What is Subword Decomposition? Traditional tokenizers choke on rare or compound words like 'cinematic', labeling them [UNKNOWN]. BPE breaks rare words into frequent, familiar sub-blocks like 'cinema' + 'tic' and 'master' + 'piece'.",
                    simpleInput: "The 11 word chunks.",
                    simpleProcessing: "Compares words against 30,000 learned vocabulary rules, keeping common words whole and breaking compound words into morphemes.",
                    simpleOutput: "13 subword units: ['What', 'an', 'absolute', 'master', 'piece', 'of', 'cinema', 'tic', 'failure', '.', 'Brilliant', 'writing', '!'].",
                    impact: "100% vocabulary coverage with zero missing words. The neural network never encounters an [UNKNOWN] token.",
                    math: {
                        formula: "Subwords = GreedyMerge(Tokens, Vocabulary_30K)",
                        inputTensor: "11 lexical fragments",
                        operation: "Iterative subword pair merge rule lookup",
                        outputTensor: "13 subword units (0 Out-Of-Vocabulary fallback tokens; 100% vocabulary coverage)"
                    },
                    input: "Candidate lexical fragments",
                    operation: "Lookup against 30,000 pre-trained BPE merge rules. Common words stay whole; compound/rare words decomposed into frequent morphemes: 'masterpiece' -> 'master' + 'piece', 'cinematic' -> 'cinema' + 'tic'.",
                    output: "13 subword units (0 Out-Of-Vocabulary fallback tokens; 100% vocabulary coverage):\n['What', 'an', 'absolute', 'master', 'piece', 'of', 'cinema', 'tic', 'failure', '.', 'Brilliant', 'writing', '!']"
                },
                {
                    stage: "Token ID Mapping & Uniform Padding",
                    concept: "What is Token ID Mapping? Neural networks cannot read letters; they only understand numbers. Each subword is looked up in a dictionary to get an integer ID, and padded with zeros to make all inputs exactly 200 numbers long.",
                    simpleInput: "The 13 subword units.",
                    simpleProcessing: "Replaces each subword with its dictionary integer ID and right-pads with zeros up to length 200.",
                    simpleOutput: "A 200-number list starting with [124, 18, 542, 5421, ...] followed by 187 padding zeros.",
                    impact: "Allows the GPU to process batches of sentences simultaneously at uniform fixed tensor dimensions.",
                    math: {
                        formula: "IDs = Pad_Right([VocabIndex(w) for w in Subwords], target_len=200, pad_id=0)",
                        inputTensor: "13 subwords",
                        operation: "Vocabulary dictionary hash lookup and zero-padding",
                        outputTensor: "Int64 Tensor (batch_size=1, seq_len=200):\n[124, 18, 542, 5421, 891, 14, 8912, 1045, 1923, 28, 4312, 1820, 2, 0, 0, ... 0]\n(Exact sequence: 13 active tokens + 187 zero-padding units)"
                    },
                    input: "13 subword tokens",
                    operation: "Map each subword to integer index in 30,000 vocabulary dictionary; right-pad with ID 0 up to uniform fixed length 200.",
                    output: "Int64 Tensor (batch_size=1, seq_len=200):\n[124, 18, 542, 5421, 891, 14, 8912, 1045, 1923, 28, 4312, 1820, 2, 0, 0, ... 0]\n(Exact sequence: 13 active tokens + 187 zero-padding units)"
                },
                {
                    stage: "Binary Attention Mask Generation",
                    concept: "What is an Attention Mask? Because we added 187 padding zeros, we must tell the AI which numbers are real words and which numbers are meaningless padding. The mask is a list of 1s (real word) and 0s (ignore).",
                    simpleInput: "The 200-number ID list.",
                    simpleProcessing: "Sets positions with real words to 1 and padded positions to 0.",
                    simpleOutput: "A binary mask list: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ... 0].",
                    impact: "Guarantees that downstream attention layers ignore padding zeros completely and never let filler distort sentiment scores.",
                    math: {
                        formula: "Mask_t = 1 if token_id_t != 0 else 0",
                        inputTensor: "Int64 Tensor (1, 200)",
                        operation: "Inequality boolean broadcasting",
                        outputTensor: "Binary Attention Mask Tensor (1, 200) Int64:\n[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ... 0]\n(Positions 13-199 masked with -1e9 in downstream Multi-Head Attention)"
                    },
                    input: "Padded ID array (1, 200)",
                    operation: "Evaluate index condition (token_id != 0); assign Boolean 1 to real tokens and 0 to padded positions.",
                    output: "Binary Attention Mask Tensor (1, 200) Int64:\n[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ... 0]\n(Positions 13-199 masked with -1e9 in downstream Multi-Head Attention)"
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 210" width="100%" height="210" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="15" width="160" height="75" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="90" y="42" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">RAW INPUT TEXT</text>
  <text x="90" y="62" font-family="'Space Mono',monospace" font-size="9" fill="#64748b" text-anchor="middle">"cinematic failure"</text>

  <line x1="170" y1="52" x2="205" y2="52" stroke="#4f46e5" stroke-width="2"/>

  <rect x="205" y="15" width="160" height="75" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="285" y="42" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">BYTE PRE-SPLIT</text>
  <text x="285" y="62" font-family="'Space Mono',monospace" font-size="9" fill="#64748b" text-anchor="middle">['cinema','tic','failure']</text>

  <line x1="365" y1="52" x2="400" y2="52" stroke="#4f46e5" stroke-width="2"/>

  <rect x="400" y="15" width="165" height="75" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="482" y="38" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#4338ca" text-anchor="middle">30K BPE VOCAB</text>
  <text x="482" y="56" font-family="'Space Mono',monospace" font-size="9" fill="#4f46e5" text-anchor="middle">Merge Table Lookup</text>
  <text x="482" y="72" font-family="'Space Mono',monospace" font-size="9" fill="#64748b" text-anchor="middle">ID Mapping</text>

  <line x1="565" y1="52" x2="600" y2="52" stroke="#4f46e5" stroke-width="2"/>

  <rect x="600" y="15" width="150" height="75" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
  <text x="675" y="38" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#15803d" text-anchor="middle">OUTPUT TENSORS</text>
  <text x="675" y="56" font-family="'Space Mono',monospace" font-size="8.5" fill="#166534" text-anchor="middle">IDs: (B, 200) Int64</text>
  <text x="675" y="72" font-family="'Space Mono',monospace" font-size="8.5" fill="#166534" text-anchor="middle">Mask: (B, 200) Bool</text>

  <rect x="10" y="115" width="740" height="80" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.2"/>
  <text x="25" y="138" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#0f172a">SUBWORD DECOMPOSITION ADVANTAGE</text>
  <text x="25" y="158" font-family="'Plus Jakarta Sans',sans-serif" font-size="10" fill="#475569">Traditional Tokenizer: "cinematic" -> [UNK] (Out of Vocabulary)</text>
  <text x="25" y="176" font-family="'Plus Jakarta Sans',sans-serif" font-size="10" fill="#4338ca" font-weight="600">dl Rust BPE Tokenizer: "cinematic" -> ["cinema" (ID 8912), "tic" (ID 1045)] (Zero Information Loss)</text>
</svg>`,
        relatedComponents: ["glove_embeddings", "whitespace_tokenizer"]
    },

    glove_embeddings: {
        id: "glove_embeddings",
        title: "Stanford GloVe 300-Dimensional Vector Embeddings",
        shortName: "GloVe 300d",
        category: "Semantic Vector Space",
        models: ["Model V2 (Multi-Task)", "Model V3 (Flagship)"],
        parameters: "9,000,000 Parameters (30,000 Vocab x 300 Dim)",
        inputShape: "Tensor (batch_size, 200) Int64 Token IDs",
        outputShape: "Tensor (batch_size, 200, 300) Float32 Continuous Vectors",
        badge: "Pre-Trained 6B Corpus",
        summary: "Continuous vector projection initialized from Stanford GloVe 6B corpus, encoding semantic geometry, polarity vectors, and contextual associations.",
        beginnerIntuition: "Think of GloVe like a 300-dimensional GPS map of human language. In this semantic coordinate system, words with related meanings live in the exact same neighborhood. The vector for 'masterpiece' is located right next to 'brilliant', while 'failure' is situated on the opposite side of the world.",
        whatIsIt: `Global Vectors for Word Representation (GloVe) is an unsupervised geometric algorithm developed by Pennington et al. at Stanford. Unlike one-hot sparse encodings where every word is an isolated orthogonal dimension, GloVe projects words into a dense 300-dimensional continuous vector space. In this space, vector distance and cosine angle mirror semantic relationships: words with similar meaning cluster together, and vector arithmetic reflects relational concepts (e.g., vector("king") - vector("man") + vector("woman") ≈ vector("queen")).`,
        howIsItTrained: `GloVe is pre-trained by minimizing a global log-bilinear objective over a word-word co-occurrence matrix X:
J = sum_{i,j=1}^{V} f(X_ij) * (w_i^T * w_j_tilde + b_i + b_j_tilde - log(X_ij))^2
where f(X_ij) is a weighting function that prevents very frequent stop-words from dominating the loss. In dl:
1. Stanford 6B Token GloVe 300d weights are loaded.
2. For BPE subwords present directly in GloVe, exact 300d vectors are assigned.
3. For composite subwords, character n-gram average pooling is applied.
4. During initial training epochs, embeddings are frozen to preserve semantic geometry, then fine-tuned with a low learning rate (5e-5) during end-to-end multi-task optimization.`,
        dataFed: `Pre-trained on 6 billion tokens harvested from:
- Wikipedia 2014 Dump (comprehensive factual descriptions, formal vocabulary)
- Gigaword 5 (massive international news archive, narrative prose)
Total vocabulary coverage exceeds 400,000 uncased words mapped down to our 30,000 BPE vocabulary.`,
        architectureRole: `The embedding layer transforms discrete integer tokens into continuous spatial activations suitable for gradient-based linear algebra in PyTorch. The 300-dimensional representation provides rich semantic context (e.g., distinguishing whether "brilliant" carries positive valence or contrast) before sequential CNN and BiLSTM processing begins.`,
        ablationStudy: `What happens if pre-trained GloVe Embeddings are absent?
- Random Embeddings Failure: In Model V1, embeddings are initialized with uniform random noise U(-0.05, +0.05). On a modest 230K dataset, random embeddings must learn English from scratch, leading to severe overfitting.
- Convergence Speed: In our ablation benchmarks, pre-trained GloVe accelerated training convergence by 3.5x (reaching 88% validation accuracy in epoch 2 vs epoch 7 for random initialization).
- Generalization: GloVe boosted test sentiment accuracy by +3.4% and sarcasm F1 by +4.1% on unseen movie reviews.`,
        inputOutputSpec: {
            inputType: "torch.LongTensor (Int64)",
            inputDimensions: "(batch_size, seq_len) = (B, 200)",
            outputType: "torch.FloatTensor (Float32)",
            outputDimensions: "(batch_size, seq_len, embed_dim) = (B, 200, 300)",
            parameters: "30,000 x 300 = 9,000,000 trainable float32 parameters"
        },
        traversalExample: {
            "sampleText": "masterpiece vs failure",
            "steps": [
                {
                    "stage": "Input Token ID Indexing",
                    "concept": "What is Token Indexing? Neural networks store word meanings in a massive lookup table like an encyclopedia. Token indexing takes the integer ID of each subword and finds its exact catalog address in the table.",
                    "simpleInput": "Token IDs for words: [5421 ('master'), 891 ('piece'), 1923 ('failure')].",
                    "simpleProcessing": "Looks up the memory row addresses in the 30,000-word GloVe encyclopedia table.",
                    "simpleOutput": "Exact row address pointers pointing to 300-number concept descriptions.",
                    "impact": "Connects raw integer IDs to rich pre-trained linguistic knowledge learned from billions of web words.",
                    "math": {
                        "formula": "addr_t = HashIndex(token_id_t, E)",
                        "inputTensor": "Int64 Tensor IDs (1, 200)",
                        "operation": "Row index pointer extraction from static weight table E in R^(30000 x 300)",
                        "outputTensor": "200 memory pointers to R^300 row vectors"
                    },
                    "input": "Token IDs [5421 ('master'), 891 ('piece'), 1923 ('failure')]",
                    "operation": "Gather memory row addresses from frozen weight table E in R^(30000 x 300).",
                    "output": "Discrete memory row pointers:\nRow 5421 ('master') -> Float32 vector in R^300\nRow 891 ('piece') -> Float32 vector in R^300\nRow 1923 ('failure') -> Float32 vector in R^300"
                },
                {
                    "stage": "Vector Row Lookup Gathering",
                    "concept": "What is Vector Gathering? Words with similar meanings have similar coordinates in 300-dimensional space. Gathering pulls out the 300 numbers representing each word's nuances, tone, and grammar.",
                    "simpleInput": "Memory pointers for all 200 token positions.",
                    "simpleProcessing": "Fetches the 300 numbers for each word from the table and stacks them into a sequence matrix.",
                    "simpleOutput": "A matrix where each word is represented by 300 descriptive numbers (e.g., 'masterpiece' and 'failure' have distinct coordinate patterns).",
                    "impact": "Transforms bare ID numbers into geometric concepts where emotional tone is encoded as spatial direction.",
                    "math": {
                        "formula": "x_t = E[token_id_t], for t in [0, 199]",
                        "inputTensor": "Tensor gather addresses (1, 200)",
                        "operation": "Dense embedding slice extraction: x_t in R^300",
                        "outputTensor": "Dense Float32 Tensor (batch_size=1, seq_len=200, embed_dim=300):\n'masterpiece': [+0.312, -0.481, +0.104, +0.655, ... ||v||_2 = 5.21]\n'failure': [-0.518, +0.229, -0.380, +0.412, ... ||v||_2 = 4.88]\nPadding tokens (ID 0): [0.000, 0.000, ... 0.000]"
                    },
                    "input": "Addresses for all 200 token positions",
                    "operation": "Tensor gather: x_t = E[token_id_t], retrieving pre-trained distributional semantic vectors.",
                    "output": "Dense Float32 Tensor (batch_size=1, seq_len=200, embed_dim=300):\n'masterpiece': [+0.312, -0.481, +0.104, +0.655, ... ||v||_2 = 5.21]\n'failure': [-0.518, +0.229, -0.380, +0.412, ... ||v||_2 = 4.88]\nPadding tokens (ID 0): [0.000, 0.000, ... 0.000]"
                },
                {
                    "stage": "Semantic Geometry Cosine Verification",
                    "concept": "What is Cosine Geometry? In 300-dimensional space, the angle between two word vectors reveals how closely related they are. Cosine similarity measures whether words point in the same direction (+1.0) or opposite directions (-1.0).",
                    "simpleInput": "Extracted 300-number vectors for praise words and criticism words.",
                    "simpleProcessing": "Calculates the directional cosine angle between praise ('masterpiece') and fault ('failure').",
                    "simpleOutput": "A clear semantic contrast: 'masterpiece' matches 'brilliant' (+0.76), but strongly clashes with 'failure' (-0.18).",
                    "impact": "Gives the neural network a geometric foundation to immediately spot clashing words that signal sarcasm.",
                    "math": {
                        "formula": "cos(u, v) = (u . v) / (||u||_2 * ||v||_2)",
                        "inputTensor": "300d vectors for u and v",
                        "operation": "Vector dot product divided by L2 norms",
                        "outputTensor": "Spatial alignment in 300d vector space:\ncos('masterpiece', 'brilliant') = +0.76 (Strong positive semantic cluster)\ncos('failure', 'disaster') = +0.82 (Strong negative semantic cluster)\ncos('masterpiece', 'failure') = -0.18 (Opposing affective vectors)"
                    },
                    "input": "Extracted 300d GloVe vectors",
                    "operation": "Compute pairwise cosine similarity: cos(u, v) = (u . v) / (||u|| * ||v||)",
                    "output": "Spatial alignment in 300d vector space:\ncos('masterpiece', 'brilliant') = +0.76 (Strong positive semantic cluster)\ncos('failure', 'disaster') = +0.82 (Strong negative semantic cluster)\ncos('masterpiece', 'failure') = -0.18 (Opposing affective vectors)"
                },
                {
                    "stage": "Spatial Channel Dropout (p = 0.2)",
                    "concept": "What is Channel Dropout? During training, a model might become lazy and rely too much on just a few numbers. Dropout randomly turns off 20% of the numbers so the network learns to use all 300 dimensions.",
                    "simpleInput": "The complete (200 words x 300 numbers) matrix.",
                    "simpleProcessing": "Randomly shuts off 60 out of 300 channels and scales the remaining channels by 1.25x to keep the total energy balanced.",
                    "simpleOutput": "A robust matrix ready for convolution, preventing the model from memorizing specific training phrases.",
                    "impact": "Prevents overfitting and forces the network to generalize across unfamiliar movie reviews.",
                    "math": {
                        "formula": "x_drop = (m * x) / (1 - p), where m ~ Bernoulli(1 - p)",
                        "inputTensor": "Float32 Tensor (1, 200, 300)",
                        "operation": "Spatial feature channel masking at probability p = 0.20",
                        "outputTensor": "Regularized Float32 Tensor (1, 200, 300):\n60 of 300 channels zeroed out for regularized feature diversity; active channels scaled by 1.25x (mean = -0.008, variance = 1.021) ready for 1D convolution."
                    },
                    "input": "Embedding Tensor (1, 200, 300)",
                    "operation": "Randomly zero out 20% of entire 300d feature channels across time; scale active channels by 1.25x.",
                    "output": "Regularized Float32 Tensor (1, 200, 300):\n60 of 300 channels zeroed out for regularized feature diversity; active channels scaled by 1.25x (mean = -0.008, variance = 1.021) ready for 1D convolution."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 190" width="100%" height="190" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="25" width="150" height="65" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="95" y="52" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">TOKEN IDS</text>
  <text x="95" y="70" font-family="'Space Mono',monospace" font-size="9" fill="#64748b" text-anchor="middle">[5421, 1923, ...]</text>

  <line x1="170" y1="57" x2="220" y2="57" stroke="#4f46e5" stroke-width="2"/>

  <rect x="220" y="15" width="230" height="85" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="335" y="42" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#4338ca" text-anchor="middle">GLOVE WEIGHT MATRIX</text>
  <text x="335" y="60" font-family="'Space Mono',monospace" font-size="9" fill="#4f46e5" text-anchor="middle">Matrix W: 30,000 x 300</text>
  <text x="335" y="78" font-family="'Space Mono',monospace" font-size="9" fill="#64748b" text-anchor="middle">9,000,000 Pretrained Weights</text>

  <line x1="450" y1="57" x2="500" y2="57" stroke="#4f46e5" stroke-width="2"/>

  <rect x="500" y="20" width="240" height="75" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
  <text x="620" y="48" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#15803d" text-anchor="middle">DENSE TENSOR</text>
  <text x="620" y="68" font-family="'Space Mono',monospace" font-size="9" fill="#166534" text-anchor="middle">Shape: (batch, 200, 300) Float32</text>

  <rect x="20" y="115" width="720" height="60" rx="8" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.2"/>
  <text x="35" y="138" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#0f172a">PRETRAINED SEMANTIC TOPOLOGY</text>
  <text x="35" y="156" font-family="'Plus Jakarta Sans',sans-serif" font-size="9.5" fill="#475569">Captures foundational relationships before training begins: cos("brilliant", "masterpiece") = +0.76</text>
</svg>`,
        relatedComponents: ["bpe_tokenizer", "multiscale_cnn"]
    },

    multiscale_cnn: {
        id: "multiscale_cnn",
        title: "4-Scale Parallel 1D Convolutions with Mish Activation",
        shortName: "Multi-Scale CNN",
        category: "Local N-Gram Feature Extraction",
        models: ["Model V2 (Multi-Task)", "Model V3 (Flagship)"],
        parameters: "326,656 Parameters (4 Parallel Conv1D Banks + BatchNorm1d)",
        inputShape: "Tensor (batch_size, 300, 200) Transposed Continuous Embeddings",
        outputShape: "Tensor (batch_size, 256, 200) Multi-Scale N-Gram Features",
        badge: "Inception-Style Parallel CNN",
        summary: "Parallel 1D convolution banks with kernel sizes k=2,3,5,7 capturing localized bigram, trigram, and clausal grammatical patterns simultaneously.",
        beginnerIntuition: "Think of Multi-Scale CNN as examining text through four different magnifying glasses at once. One lens looks at 2 words at a time (e.g. 'cinematic failure'), another looks at 3 words, and another looks at 5 words. This allows the neural network to catch localized sarcastic phrases instantly without reading the entire sentence sequentially.",
        whatIsIt: `The Multi-Scale CNN module adapts computer vision Inception blocks to natural language processing. Instead of forcing all sequence features through a single kernel filter, dl employs 4 parallel 1D convolutional layers operating side-by-side with kernel sizes k in {2, 3, 5, 7}. Each kernel specializes in extracting localized linguistic features at a distinct syntactic scale:
- k = 2 captures bigram modifiers ("terrible movie", "brilliant writing")
- k = 3 captures trigrams ("not very good", "waste of time")
- k = 5 captures clausal phrases ("one of the worst ever")
- k = 7 captures extended sarcastic setups ("what a masterpiece of failure")`,
        howIsItTrained: `Trained end-to-end via backpropagation using AdamW optimizer with Mish non-linear activation:
f(x) = x * tanh(ln(1 + e^x))
Mish avoids the 'dying ReLU' problem by maintaining smooth, non-monotonic gradients for negative activations, essential for subtle sarcasm features. Outputs from each kernel are padded to preserve temporal length (seq_len = 200), concatenated into 256 channels, normalized with 1D Batch Normalization (affine=True), and passed through Spatial Dropout (p = 0.25).`,
        dataFed: `Receives continuous 300-dimensional word representations generated by the GloVe embedding layer. Operates across all 200 sequence time steps simultaneously.`,
        architectureRole: `Extracts position-invariant n-gram cues before sequential BiLSTM processing. Sarcasm often hinges on tight local contradictions (e.g., placing the positive word "masterpiece" adjacent to "cinematic failure"). The 4 parallel kernels detect these localized collisions instantly, feeding rich n-gram feature maps into the recurrent layers.`,
        ablationStudy: `What happens if Multi-Scale Convolutions are absent?
- Single Kernel Degradation: In Model V1 (which uses standard k=3,5,7 with 96 filters), localized bigrams (k=2) like "mock praise" are missed.
- No-CNN Ablation: Feeding embeddings directly into a BiLSTM without CNN pre-filtering reduces validation accuracy by 3.1% and slows training convergence by 40%, because the recurrent cells are forced to perform both localized phrase detection and global sequence tracking simultaneously.`,
        inputOutputSpec: {
            inputType: "torch.FloatTensor (Float32)",
            inputDimensions: "(batch_size, embed_dim, seq_len) = (B, 300, 200)",
            outputType: "torch.FloatTensor (Float32)",
            outputDimensions: "(batch_size, total_filters, seq_len) = (B, 256, 200)",
            kernels: "k=2 (64 filters), k=3 (64 filters), k=5 (64 filters), k=7 (64 filters)",
            parameters: "Conv1d: (300*2*64 + 300*3*64 + 300*5*64 + 300*7*64) + 256 biases = 326,656 params"
        },
        traversalExample: {
            "sampleText": "masterpiece of cinematic failure",
            "steps": [
                {
                    "stage": "Transposed Input Preparation",
                    "concept": "What is Transposition? 1D Convolution operates like a scanner sliding across time. In PyTorch, the scanner expects the 300 feature channels to be stacked vertically so it can scan along the 200 words horizontally.",
                    "simpleInput": "A sequence tensor organized as (1 sentence, 200 words, 300 features).",
                    "simpleProcessing": "Swaps the dimensions into (1 sentence, 300 features, 200 words) to match the GPU's convolution layout.",
                    "simpleOutput": "A transposed tensor ready for simultaneous multi-filter scanning.",
                    "impact": "Allows the parallel convolution banks to run at maximum GPU speed with contiguous memory access.",
                    "math": {
                        "formula": "X_transposed = permute(X, (0, 2, 1))",
                        "inputTensor": "Embedding tensor (batch_size=1, seq_len=200, embed_dim=300)",
                        "operation": "Permute axes to (B, C, T) = (1, 300, 200) to match PyTorch Conv1d memory layout.",
                        "outputTensor": "Float32 Tensor (batch_size=1, in_channels=300, length=200):\nContiguous memory layout with 300 parallel input channels across 200 time steps. Active sequence spans steps 0-12; steps 13-199 hold zeroed pad channels."
                    },
                    "input": "Embedding tensor (batch_size=1, seq_len=200, embed_dim=300)",
                    "operation": "Permute axes to (B, C, T) = (1, 300, 200) to match PyTorch Conv1d memory layout.",
                    "output": "Float32 Tensor (batch_size=1, in_channels=300, length=200):\nContiguous memory layout with 300 parallel input channels across 200 time steps. Active sequence spans steps 0-12; steps 13-199 hold zeroed pad channels."
                },
                {
                    "stage": "Parallel Conv1D Banks (k=2, 3, 5, 7)",
                    "concept": "What are Parallel Filter Banks? Sarcasm often hides in short 2-word idioms ('yeah right'), medium 3-word phrases ('cinematic failure'), or long 7-word exaggerated praises ('What an absolute masterpiece of cinematic failure'). Four filter banks scan the sentence simultaneously at different window sizes.",
                    "simpleInput": "300-channel sequence fed into 4 different scanner sizes: 2 words, 3 words, 5 words, and 7 words.",
                    "simpleProcessing": "Slides filters over the text, computing phrase patterns and applying smooth Mish activation.",
                    "simpleOutput": "Four distinct feature maps (64 channels each) capturing subword rhymes, short phrases, and full sarcastic clauses.",
                    "impact": "Catches irony whether it appears as a quick 2-word jab or an elaborate 7-word mock compliment.",
                    "math": {
                        "formula": "F_k = Mish(Conv1D_k(X)) = Conv1D_k(X) * tanh(ln(1 + exp(Conv1D_k(X))))",
                        "inputTensor": "300-channel sequence fed into 4 parallel kernel banks (64 filters each)",
                        "operation": "Apply 1D convolution with stride=1, same-padding, followed by Mish non-linearity: y = x * tanh(ln(1 + e^x)).",
                        "outputTensor": "4 parallel feature map tensors (64 channels each):\n- k=2 (bigram motifs): (1, 64, 200) Float32\n- k=3 (trigrams 'cinematic failure'): (1, 64, 200) Float32\n- k=5 (5-grams 'masterpiece of cinematic failure'): (1, 64, 200) Float32\n- k=7 (7-gram clashing phrase context): (1, 64, 200) Float32"
                    },
                    "input": "300-channel sequence fed into 4 parallel kernel banks (64 filters each)",
                    "operation": "Apply 1D convolution with stride=1, same-padding, followed by Mish non-linearity: y = x * tanh(ln(1 + e^x)).",
                    "output": "4 parallel feature map tensors (64 channels each):\n- k=2 (bigram motifs): (1, 64, 200) Float32\n- k=3 (trigrams 'cinematic failure'): (1, 64, 200) Float32\n- k=5 (5-grams 'masterpiece of cinematic failure'): (1, 64, 200) Float32\n- k=7 (7-gram clashing phrase context): (1, 64, 200) Float32"
                },
                {
                    "stage": "Channel Dimension Concatenation",
                    "concept": "What is Channel Concatenation? Merges the 4 separate scanner results (64 channels each) side-by-side into a single rich stream of 256 feature channels.",
                    "simpleInput": "Four separate feature maps of 64 channels each.",
                    "simpleProcessing": "Glues all 4 sets of channels together along the feature axis.",
                    "simpleOutput": "A unified map with 256 channels per word, combining both short-range and long-range phrase patterns.",
                    "impact": "Provides downstream recurrent layers with simultaneous access to local word idioms and broader sarcastic tone.",
                    "math": {
                        "formula": "F_unified = concat([f_k2, f_k3, f_k5, f_k7], dim=1)",
                        "inputTensor": "Four (1, 64, 200) feature maps",
                        "operation": "Concatenate along channel dimension: cat([f_k2, f_k3, f_k5, f_k7], dim=1).",
                        "outputTensor": "Unified Feature Map (batch_size=1, channels=256, length=200) Float32:\nChannels 0-63 capture short subwords; 64-127 capture trigram mock-praise; 128-191 capture 5-word phrases; 192-255 capture long-range contextual clashes."
                    },
                    "input": "Four (1, 64, 200) feature maps",
                    "operation": "Concatenate along channel dimension: cat([f_k2, f_k3, f_k5, f_k7], dim=1).",
                    "output": "Unified Feature Map (batch_size=1, channels=256, length=200) Float32:\nChannels 0-63 capture short subwords; 64-127 capture trigram mock-praise; 128-191 capture 5-word phrases; 192-255 capture long-range contextual clashes."
                },
                {
                    "stage": "Batch Normalization & Dropout Regularization",
                    "concept": "What is Batch Normalization? When numbers pass through convolutions, values can become too large or drift. Normalization centers the numbers around 0, stabilizing the training so the model learns reliably.",
                    "simpleInput": "Raw 256-channel convolutional map.",
                    "simpleProcessing": "Normalizes the activations (zero mean, unit variance), drops 25% for robustness, and swaps back to sequence format.",
                    "simpleOutput": "A clean, stable (1 sentence, 200 words, 256 features) tensor ready for BiLSTM.",
                    "impact": "Guarantees smooth gradient flow and prevents internal covariate shift during multi-task training.",
                    "math": {
                        "formula": "Y = Dropout_0.25(permute(BatchNorm1d(F_unified), (0, 2, 1)))",
                        "inputTensor": "(1, 256, 200) raw convolutional feature map",
                        "operation": "Apply BatchNorm1d (learnable gamma, beta), apply 25% dropout, and transpose back to (B, T, C).",
                        "outputTensor": "Normalized Float32 Tensor (batch_size=1, seq_len=200, channels=256):\nActivation statistics: mean = -0.014, variance = 0.992. Local n-gram syntactic motifs stabilized and conditioned for recurrent BiLSTM consumption."
                    },
                    "input": "(1, 256, 200) raw convolutional feature map",
                    "operation": "Apply BatchNorm1d (learnable gamma, beta), apply 25% dropout, and transpose back to (B, T, C).",
                    "output": "Normalized Float32 Tensor (batch_size=1, seq_len=200, channels=256):\nActivation statistics: mean = -0.014, variance = 0.992. Local n-gram syntactic motifs stabilized and conditioned for recurrent BiLSTM consumption."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 210" width="100%" height="210" xmlns="http://www.w3.org/2000/svg">
  <rect x="15" y="70" width="130" height="70" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="80" y="100" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="middle">EMBEDDINGS</text>
  <text x="80" y="118" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">(B, 300, 200)</text>

  <path d="M 145 105 L 200 35 L 230 35" fill="none" stroke="#4f46e5" stroke-width="1.8"/>
  <path d="M 145 105 L 200 80 L 230 80" fill="none" stroke="#4f46e5" stroke-width="1.8"/>
  <path d="M 145 105 L 200 130 L 230 130" fill="none" stroke="#4f46e5" stroke-width="1.8"/>
  <path d="M 145 105 L 200 175 L 230 175" fill="none" stroke="#4f46e5" stroke-width="1.8"/>

  <rect x="230" y="15" width="170" height="40" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="1.2"/>
  <text x="315" y="38" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#4338ca" text-anchor="middle">Conv1D (k=2) -> 64 filters</text>

  <rect x="230" y="60" width="170" height="40" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="1.2"/>
  <text x="315" y="83" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#4338ca" text-anchor="middle">Conv1D (k=3) -> 64 filters</text>

  <rect x="230" y="110" width="170" height="40" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="1.2"/>
  <text x="315" y="133" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#4338ca" text-anchor="middle">Conv1D (k=5) -> 64 filters</text>

  <rect x="230" y="155" width="170" height="40" rx="6" fill="#eef2ff" stroke="#6366f1" stroke-width="1.2"/>
  <text x="315" y="178" font-family="'Space Mono',monospace" font-size="9" font-weight="700" fill="#4338ca" text-anchor="middle">Conv1D (k=7) -> 64 filters</text>

  <path d="M 400 35 L 430 35 L 470 105" fill="none" stroke="#4f46e5" stroke-width="1.8"/>
  <path d="M 400 80 L 430 80 L 470 105" fill="none" stroke="#4f46e5" stroke-width="1.8"/>
  <path d="M 400 130 L 430 130 L 470 105" fill="none" stroke="#4f46e5" stroke-width="1.8"/>
  <path d="M 400 175 L 430 175 L 470 105" fill="none" stroke="#4f46e5" stroke-width="1.8"/>

  <rect x="470" y="70" width="270" height="70" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
  <text x="605" y="98" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#15803d" text-anchor="middle">CONCATENATED FEATURES</text>
  <text x="605" y="116" font-family="'Space Mono',monospace" font-size="9" fill="#166534" text-anchor="middle">Shape: (B, 256, 200) [64x4 Channels]</text>
</svg>`,
        relatedComponents: ["glove_embeddings", "bilstm"]
    },

    bilstm: {
        id: "bilstm",
        title: "Deep Bidirectional Long Short-Term Memory (BiLSTM)",
        shortName: "Deep BiLSTM",
        category: "Long-Range Temporal Context",
        models: ["Model V2 (Multi-Task)", "Model V3 (Flagship)"],
        parameters: "788,480 Parameters (2 Layers, Hidden Size 128 per direction)",
        inputShape: "Tensor (batch_size, 200, 256) Transposed CNN Feature Maps",
        outputShape: "Tensor (batch_size, 200, 256) Bidirectional Hidden States",
        badge: "2-Layer Recurrent Core",
        summary: "Bidirectional recurrent network processing sequential context forwards and backwards simultaneously, modeling narrative progression and long-range setup-punchline dependencies.",
        beginnerIntuition: "Think of BiLSTM as reading a mystery book twice: once from the first page to the end, and then backwards from the end to the beginning. When you reach a sentence like 'What a masterpiece...', you already know from the backwards reading that '...of catastrophic failure' is waiting at the end, so you are never fooled by the surface praise.",
        whatIsIt: `Bidirectional Long Short-Term Memory (BiLSTM) is a recurrent architecture designed to overcome the vanishing gradient problem when modeling sequential dependencies across long text sequences. Unlike standard forward-only LSTMs that only know what has been said in the past, a BiLSTM maintains two independent recurrent hidden layers:
1. Forward LSTM (h_t_forward): Reads left-to-right from token 1 to 200.
2. Backward LSTM (h_t_backward): Reads right-to-left from token 200 to 1.
At every time step t, the hidden vectors are concatenated: h_t = [h_t_forward ; h_t_backward], giving every word complete awareness of both its historical setup and its future consequence.`,
        howIsItTrained: `Trained via Backpropagation Through Time (BPTT) with gradient norm clipping at 1.0. Employs internal forget gates:
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
allowing the network to remember early sarcastic setups while discarding neutral narrative filler. Layer dropout of 0.2 is applied between stacked recurrent layers.`,
        dataFed: `Receives 256-dimensional feature representations produced by the Multi-Scale CNN at all 200 time steps.`,
        architectureRole: `Sarcasm and irony fundamentally depend on temporal contrast between setup and punchline. A reviewer may write three glowing sentences before delivering a devastating punchline ("...and then I woke up from that beautiful daydream into this dumpster fire"). The BiLSTM propagates context across the entire sequence so that when the attention and incongruity modules process the sentence, the setup and punchline are inextricably linked.`,
        ablationStudy: `What happens if the BiLSTM is absent?
- Loss of Temporal Order: Relying purely on CNNs or pooling ignores word order across long distances. In ablation tests without BiLSTM, the model failed on reviews longer than 35 words where sarcasm appeared at the very end.
- Accuracy Drop: Removing the BiLSTM caused a 5.7% drop in sarcasm accuracy on long IMDB reviews.`,
        inputOutputSpec: {
            inputType: "torch.FloatTensor (Float32)",
            inputDimensions: "(batch_size, seq_len, cnn_dim) = (B, 200, 256)",
            outputType: "torch.FloatTensor (Float32)",
            outputDimensions: "(batch_size, seq_len, hidden_dim*2) = (B, 200, 256)",
            parameters: "2 layers * 4 gates * (input_dim + hidden_dim + 1) * hidden_dim * 2 directions = 788,480 params"
        },
        traversalExample: {
            "sampleText": "What an absolute masterpiece of cinematic failure. Brilliant writing!",
            "steps": [
                {
                    "stage": "Forward & Backward States at t=3 ('masterpiece')",
                    "concept": "What is Bi-directional Reading? Humans read reviews forward, but context often depends on words that come later. A BiLSTM reads the sentence in two directions at once: left-to-right (past context) and right-to-left (future context).",
                    "simpleInput": "The word 'masterpiece' surrounded by 'What an absolute' in the past and 'of cinematic failure' in the future.",
                    "simpleProcessing": "Computes forward LSTM memory from the start and backward LSTM memory from the end, then combines both 128-number vectors.",
                    "simpleOutput": "A 256-number state where forward memory registers positive praise while backward memory anticipates the incoming word 'failure'.",
                    "impact": "Catches the tension of mock praise before the reader even finishes the sentence.",
                    "math": {
                        "formula": "h_3 = [h_fwd_3 ; h_bwd_3], where h_fwd = LSTM_fwd(x_0..3), h_bwd = LSTM_bwd(x_199..3)",
                        "inputTensor": "CNN feature vector x_3 in R^256 for token 'masterpiece' surrounded by 'What an absolute'",
                        "operation": "Compute forward LSTM (tokens 0->3) and backward LSTM (tokens 12->3); concatenate states: h_3 = [h_fwd_3 ; h_bwd_3].",
                        "outputTensor": "Concatenated State Vector h_3 in R^256 (L2-norm: ||h_3||_2 = 9.87):\n- h_fwd_3 (dims 0-127): L2-norm = 6.82, peak activation on positive praise (+2.41 on aesthetic mastery)\n- h_bwd_3 (dims 128-255): L2-norm = 7.14, carries anticipatory negative context (-1.94 on criticism) from processing 'failure. Brilliant writing!' in reverse"
                    },
                    "input": "CNN feature vector x_3 in R^256 for token 'masterpiece' surrounded by 'What an absolute'",
                    "operation": "Compute forward LSTM (tokens 0->3) and backward LSTM (tokens 12->3); concatenate states: h_3 = [h_fwd_3 ; h_bwd_3].",
                    "output": "Concatenated State Vector h_3 in R^256 (L2-norm: ||h_3||_2 = 9.87):\n- h_fwd_3 (dims 0-127): L2-norm = 6.82, peak activation on positive praise (+2.41 on aesthetic mastery)\n- h_bwd_3 (dims 128-255): L2-norm = 7.14, carries anticipatory negative context (-1.94 on criticism) from processing 'failure. Brilliant writing!' in reverse\nResult: Encodes surface praise while actively registering impending contextual negation."
                },
                {
                    "stage": "Recurrent Memory Cell Gating at t=6 ('failure')",
                    "concept": "What is LSTM Gating? LSTMs use forget, input, and output gates to decide what information to keep and what to overwrite. When the word 'failure' hits, the gates must update the emotional memory immediately.",
                    "simpleInput": "The negative fault word 'failure' arriving at step 6.",
                    "simpleProcessing": "Forget gate drops naive praise; input gate records the severe negative violation into internal memory cells.",
                    "simpleOutput": "A state vector dominated by negative sentiment, registering a direct clash with earlier praise words.",
                    "impact": "Stores the exact turning point where the review reveals its critical nature.",
                    "math": {
                        "formula": "c_t = f_t * c_{t-1} + i_t * c_tilde_t; h_t = o_t * tanh(c_t)",
                        "inputTensor": "Recurrent transition from h_5 ('cinematic') to h_6 ('failure')",
                        "operation": "Forget gate drops naive positive euphoria; input gate writes severe negative violation into memory cell c_6.",
                        "outputTensor": "Hidden State h_6 in R^256 (L2-norm: ||h_6||_2 = 11.45):\n- h_fwd_6: Strong negative swing (-3.12 on flaw detection)\n- State vector encodes sharp internal contradiction between t=3 praise and t=6 disaster"
                    },
                    "input": "Recurrent transition from h_5 ('cinematic') to h_6 ('failure')",
                    "operation": "Forget gate drops naive positive euphoria; input gate writes severe negative violation into memory cell c_6.",
                    "output": "Hidden State h_6 in R^256 (L2-norm: ||h_6||_2 = 11.45):\n- h_fwd_6: Strong negative swing (-3.12 on flaw detection)\n- State vector encodes sharp internal contradiction between t=3 praise and t=6 disaster"
                },
                {
                    "stage": "Complete Contextual Sequence Tensor Synthesis",
                    "concept": "What is the Full Sequence Tensor? Stacks the combined forward-and-backward understanding for all 200 positions so downstream layers can see how every word relates to every other word.",
                    "simpleInput": "Recurrent states across all 200 token steps.",
                    "simpleProcessing": "Stacks all 200 state vectors and masks out padding positions so empty space does not affect predictions.",
                    "simpleOutput": "A complete matrix (1 sentence, 200 words, 256 features) encoding deep sequential context.",
                    "impact": "Gives the Incongruity Module and Attention layers full visibility into every word's bidirectional meaning.",
                    "math": {
                        "formula": "H = [h_0, h_1, ... h_199] in R^(1 x 200 x 256)",
                        "inputTensor": "All 200 recurrent sequence step representations",
                        "operation": "Stack recurrent states across time; apply sequence mask: H = H * mask.unsqueeze(-1).",
                        "outputTensor": "Dense Context Tensor H in R^(batch_size=1, seq_len=200, hidden_dim=256) Float32:\nTokens 0-12 hold rich bidirectional syntactic and affective context; tokens 13-199 hold masked zero vectors ready for Multi-Head Attention."
                    },
                    "input": "All 200 recurrent sequence step representations",
                    "operation": "Stack recurrent states across time; apply sequence mask: H = H * mask.unsqueeze(-1).",
                    "output": "Dense Context Tensor H in R^(batch_size=1, seq_len=200, hidden_dim=256) Float32:\nTokens 0-12 hold rich bidirectional syntactic and affective context; tokens 13-199 hold masked zero vectors ready for Multi-Head Attention."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 180" width="100%" height="180" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="60" width="120" height="60" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="80" y="88" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="middle">CNN INPUT</text>
  <text x="80" y="104" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">(B, 200, 256)</text>

  <line x1="140" y1="90" x2="200" y2="90" stroke="#4f46e5" stroke-width="2"/>

  <rect x="200" y="20" width="310" height="60" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="355" y="45" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#4338ca" text-anchor="middle">FORWARD LSTM CELL (-->)</text>
  <text x="355" y="63" font-family="'Space Mono',monospace" font-size="8.5" fill="#6366f1" text-anchor="middle">Reads left-to-right (128 units)</text>

  <rect x="200" y="100" width="310" height="60" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="355" y="125" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#4338ca" text-anchor="middle">BACKWARD LSTM CELL (<--)</text>
  <text x="355" y="143" font-family="'Space Mono',monospace" font-size="8.5" fill="#6366f1" text-anchor="middle">Reads right-to-left (128 units)</text>

  <line x1="510" y1="50" x2="570" y2="90" stroke="#4f46e5" stroke-width="2"/>
  <line x1="510" y1="130" x2="570" y2="90" stroke="#4f46e5" stroke-width="2"/>

  <rect x="570" y="60" width="170" height="60" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
  <text x="655" y="88" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#15803d" text-anchor="middle">CONCATENATED h_t</text>
  <text x="655" y="104" font-family="'Space Mono',monospace" font-size="8.5" fill="#166534" text-anchor="middle">(B, 200, 256) Bidirectional</text>
</svg>`,
        relatedComponents: ["multiscale_cnn", "incongruity_module", "multihead_attention"]
    },

    incongruity_module: {
        id: "incongruity_module",
        title: "Sequence Split & Contrast Incongruity Module",
        shortName: "Incongruity Module",
        category: "Irony & Semantic Contradiction",
        models: ["Model V2 (Multi-Task)", "Model V3 (Flagship)"],
        parameters: "0 Additional Parameters (Mathematical Contrast Operator)",
        inputShape: "Tensor (batch_size, 200, 256) BiLSTM Sequence Hidden States",
        outputShape: "Delta_abs (B, 256), Hadamard_prod (B, 256), Incongruity_score (B,)",
        badge: "Custom dl Innovation",
        summary: "Splits sequence at midpoint to compute absolute disparity and Hadamard alignment between first-half setup and second-half punchline, directly quantifying tonal shift.",
        beginnerIntuition: "Think of this module like a lie detector that measures emotional contrast. It takes the average emotional temperature of the first half of the sentence and subtracts the emotional temperature of the second half. If someone starts with extreme praise and ends with harsh critique, the difference spikes to a high score, ringing the sarcasm alarm.",
        whatIsIt: `The Sequence Incongruity Module is a novel inductive bias engineered specifically for dl to detect the core linguistic mechanic of irony: semantic incongruity between premise and conclusion. Sarcastic sentences typically open with conventional praise before executing a tonal pivot to ridicule. The Incongruity Module explicitly measures this intra-sentence collision by bisecting the BiLSTM sequence at its temporal midpoint (L/2):
- h_first = mean(H[:, :midpoint, :])
- h_second = mean(H[:, midpoint:, :])
It then calculates:
1. Absolute Difference: Delta_abs = |h_first - h_second| in R^256 (magnitude of shift)
2. Hadamard Product: p_hadamard = h_first * h_second in R^256 (feature alignment/reversal)
3. Scalar Incongruity Score: S_inc = ||Delta_abs||_2 (L2 norm measuring overall semantic friction)`,
        howIsItTrained: `Requires no independent trainable weights. Instead, it provides a strong structural inductive bias that backpropagates gradients directly into the BiLSTM and CNN layers, forcing them to orient their internal feature representations such that contradictory emotional vectors produce high L2 norms.`,
        dataFed: `Receives the full (batch, 200, 256) hidden sequence states from the BiLSTM.`,
        architectureRole: `Directly feeds both the shared dense representation and the Sarcasm Head with an explicit mathematical quantification of tonal shift. It also generates the live Sequence Incongruity Score displayed in the dl web benchmark interface (e.g. 1.29 vs 0.28 for literal statements).`,
        ablationStudy: `What happens if the Incongruity Module is absent?
- Sarcasm Blindness: Standard neural networks attempt to pool the whole sequence into a single average, washing out the contrast between setup and punchline.
- Empirical Result: In ablation tests without the Incongruity Module, sarcasm detection accuracy on balanced test sets dropped by 8.5% (from 83.0% down to 74.5%), confirming that explicit contradiction modeling is vital for sarcasm classification.`,
        inputOutputSpec: {
            inputType: "torch.FloatTensor (Float32)",
            inputDimensions: "(batch_size, seq_len, hidden_dim) = (B, 200, 256)",
            outputType: "Dict with Delta_abs (B, 256), Hadamard_prod (B, 256), Magnitude (B,)",
            parameters: "0 trainable parameters (Pure tensor calculus operator)"
        },
        traversalExample: {
            "sampleText": "masterpiece vs failure contradiction",
            "steps": [
                {
                    "stage": "Sequence Splitting into Clause Halves",
                    "concept": "What is Clause Splitting? Sarcasm thrives on contrast: the first half of a review often sets up positive expectations, while the second half delivers the punchline. Splitting divides the review into opening and punchline sections.",
                    "simpleInput": "The 13-token active review sequence.",
                    "simpleProcessing": "Divides the sequence into the setup clause (tokens 0-6: 'What an absolute masterpiece...') and the punchline clause (tokens 7-12: 'failure. Brilliant writing!').",
                    "simpleOutput": "Two distinct sub-sequences representing setup versus punchline.",
                    "impact": "Isolates the two clashing halves so their emotional contradiction can be measured mathematically.",
                    "math": {
                        "formula": "H_setup = H[:, 0:split, :], H_punchline = H[:, split:len, :]",
                        "inputTensor": "Sequence tensor H (1, 13, 256)",
                        "operation": "Sequence midpoint partitioning along temporal axis: split_idx = 7.",
                        "outputTensor": "Two clause tensors:\n- First half H_1 (tokens 0-6): 'What an absolute masterpiece of'\n- Second half H_2 (tokens 7-12): 'cinematic failure. Brilliant writing!'"
                    },
                    "input": "BiLSTM sequence tensor H (1, 13, 256)",
                    "operation": "Partition sequence along temporal midpoint: split_idx = 7.",
                    "output": "Two clause tensors:\n- First half H_1 (tokens 0-6): 'What an absolute masterpiece of'\n- Second half H_2 (tokens 7-12): 'cinematic failure. Brilliant writing!'"
                },
                {
                    "stage": "Hadamard Element-Wise Interaction Tensor",
                    "concept": "What is a Hadamard Product? An element-by-element multiplication of two vectors. If both halves agree on praise, the product is positive. If one half is positive and the other is negative, the product produces strong negative discrepancy values.",
                    "simpleInput": "Mean vector of setup clause u in R^256 and punchline clause v in R^256.",
                    "simpleProcessing": "Multiplies matching features together: u * v, pinpointing where praise clashes with condemnation.",
                    "simpleOutput": "An interaction vector where negative spikes indicate direct thematic contradiction.",
                    "impact": "Extracts mathematical proof of sarcasm: high contradiction between praise and reality.",
                    "math": {
                        "formula": "M = u (x) v = [u_1 * v_1, u_2 * v_2, ... u_256 * v_256]",
                        "inputTensor": "Mean pooled clause vectors u in R^256 and v in R^256",
                        "operation": "Element-wise Hadamard product: M = u (x) v, computing dimension-by-dimension affective concordance.",
                        "outputTensor": "Interaction Vector M in R^256:\n- Positive concordance on aesthetic grammar (+1.14)\n- Severe negative discordance on sentiment features: dim 42 = (+2.15) * (-1.82) = -3.913 (Violent contradiction between 'masterpiece' and 'failure')"
                    },
                    "input": "Mean pooled clause vectors u in R^256 and v in R^256",
                    "operation": "Element-wise Hadamard product: M = u (x) v, computing dimension-by-dimension affective concordance.",
                    "output": "Interaction Vector M in R^256:\n- Positive concordance on aesthetic grammar (+1.14)\n- Severe negative discordance on sentiment features: dim 42 = (+2.15) * (-1.82) = -3.913 (Violent contradiction between 'masterpiece' and 'failure')"
                },
                {
                    "stage": "Cosine Contrast Metric Computation",
                    "concept": "What is the Incongruity Metric? Computes an overall contradiction score. If a review has zero conflict, the score is near 0. If severe sarcasm is present, the contradiction score spikes above 1.0.",
                    "simpleInput": "Setup vector u and punchline vector v.",
                    "simpleProcessing": "Measures directional divergence: 1 - cosine_similarity(u, v).",
                    "simpleOutput": "Incongruity score = 1.29 (exceeds high-conflict threshold 0.85).",
                    "impact": "Provides a single unmistakable scalar alert indicating intense irony and mock praise.",
                    "math": {
                        "formula": "Incongruity_Score = 1.0 - cos(u, v) = 1.0 - (-0.29) = 1.29",
                        "inputTensor": "Vectors u in R^256, v in R^256",
                        "operation": "Cosine alignment: cos(u, v) = (u . v) / (||u|| * ||v||) = -0.29; Incongruity = 1.0 - cos(u, v).",
                        "outputTensor": "Scalar Incongruity Score = 1.29:\nExceeds trigger threshold (0.85). Concludes the sequence contains explicit semantic dissonance."
                    },
                    "input": "Vectors u in R^256, v in R^256",
                    "operation": "Cosine alignment: cos(u, v) = (u . v) / (||u|| * ||v||) = -0.29; Incongruity = 1.0 - cos(u, v).",
                    "output": "Scalar Incongruity Score = 1.29:\nExceeds trigger threshold (0.85). Concludes the sequence contains explicit semantic dissonance."
                },
                {
                    "stage": "Multi-Scale Difference & Contradiction Fusion",
                    "concept": "What is Contradiction Fusion? Combines the difference (|u - v|), the interaction (u * v), and the raw vectors into a 128-number summary describing the exact nature of the contradiction.",
                    "simpleInput": "Difference vector, Hadamard vector, and clause vectors.",
                    "simpleProcessing": "Concatenates vectors, projects through a dense layer with Mish activation, and normalizes.",
                    "simpleOutput": "A 128-number contradiction vector describing the clash.",
                    "impact": "Delivers explicit evidence of sarcasm directly to the gated polarity inversion module.",
                    "math": {
                        "formula": "inc = LayerNorm(Mish(W_inc * [u ; v ; |u - v| ; u (x) v] + b))",
                        "inputTensor": "Concatenated interaction components [u ; v ; |u - v| ; u (x) v] in R^1024",
                        "operation": "Dense projection to R^128 + Mish activation + LayerNorm.",
                        "outputTensor": "Incongruity Feature Vector inc in R^128 (L2-norm: ||inc||_2 = 6.42):\nEncodes structured discrepancy profile ready for multi-head attention and gated inversion."
                    },
                    "input": "Concatenated interaction components [u ; v ; |u - v| ; u (x) v] in R^1024",
                    "operation": "Dense projection to R^128 + Mish activation + LayerNorm.",
                    "output": "Incongruity Feature Vector inc in R^128 (L2-norm: ||inc||_2 = 6.42):\nEncodes structured discrepancy profile ready for multi-head attention and gated inversion."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 200" width="100%" height="200" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="25" width="220" height="60" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="130" y="52" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#4338ca" text-anchor="middle">FIRST HALF h_first</text>
  <text x="130" y="70" font-family="'Space Mono',monospace" font-size="8.5" fill="#6366f1" text-anchor="middle">"What an absolute masterpiece"</text>

  <rect x="20" y="115" width="220" height="60" rx="8" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5"/>
  <text x="130" y="142" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#b91c1c" text-anchor="middle">SECOND HALF h_second</text>
  <text x="130" y="160" font-family="'Space Mono',monospace" font-size="8.5" fill="#dc2626" text-anchor="middle">"of cinematic failure. Terrible."</text>

  <line x1="240" y1="55" x2="330" y2="90" stroke="#4f46e5" stroke-width="2"/>
  <line x1="240" y1="145" x2="330" y2="110" stroke="#4f46e5" stroke-width="2"/>

  <rect x="330" y="70" width="190" height="65" rx="8" fill="#ffffff" stroke="#0f172a" stroke-width="1.5"/>
  <text x="425" y="96" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#0f172a" text-anchor="middle">CONTRAST OPERATOR</text>
  <text x="425" y="114" font-family="'Space Mono',monospace" font-size="8.5" fill="#475569" text-anchor="middle">|h1 - h2| and h1 * h2</text>

  <line x1="520" y1="102" x2="570" y2="102" stroke="#4f46e5" stroke-width="2"/>

  <rect x="570" y="65" width="170" height="75" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
  <text x="655" y="95" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#15803d" text-anchor="middle">INCONGRUITY SCORE</text>
  <text x="655" y="115" font-family="'Space Mono',monospace" font-size="12" font-weight="800" fill="#166534" text-anchor="middle">Score: 1.73</text>
</svg>`,
        relatedComponents: ["bilstm", "gated_polarity_inversion", "multihead_attention"]
    },

    multihead_attention: {
        id: "multihead_attention",
        title: "4-Head Scaled Dot-Product Self-Attention",
        shortName: "Multi-Head Attention",
        category: "Dynamic Global Weighting & Explainability",
        models: ["Model V2 (Multi-Task)", "Model V3 (Flagship)"],
        parameters: "263,168 Parameters (Q, K, V Projections + Out Projection)",
        inputShape: "Tensor (batch_size, 200, 256) BiLSTM Sequence Representations",
        outputShape: "Context Vector (batch_size, 256) + Attention Weights (batch_size, 4, 200, 200)",
        badge: "Transformer-Grade Attention",
        summary: "Multi-head attention projecting sequences into 4 distinct metric subspaces, dynamically weighting crucial sentiment/sarcasm tokens and generating explainable subword heatmaps.",
        beginnerIntuition: "Think of Multi-Head Attention like a panel of four specialized human editors reading a manuscript in parallel. Editor 1 watches for emotional punctuation ('!'), Editor 2 tracks sarcastic word pairs ('brilliant failure'), Editor 3 monitors negations ('never', 'not'), and Editor 4 highlights the most important words for the final verdict.",
        whatIsIt: `Multi-Head Self-Attention is the foundational mechanism introduced by Vaswani et al. in the Transformer architecture. Rather than performing a single static pooling across the sequence, dl projects the 256-dimensional BiLSTM output into 4 parallel query (Q), key (K), and value (V) subspaces of dimension d_k = 64:
head_i = softmax((Q_i * K_i^T) / sqrt(d_k) + M_pad) * V_i
MultiHead(Q,K,V) = Concat(head_1, head_2, head_3, head_4) * W_O
Each head specializes in tracking a distinct linguistic facet:
- Head 1 attends to modifier-noun relationships ("cinematic" -> "failure")
- Head 2 attends to emotional punctuation ("!")
- Head 3 attends to negative operators ("not", "never")
- Head 4 computes global token importance across the sentence.`,
        howIsItTrained: `Trained end-to-end with Scaled Dot-Product scaling (1 / sqrt(64) = 0.125) to prevent softmax saturation into extreme regions with tiny gradients. Masked softmax applies -1e4 to padding positions to guarantee zero gradient flow through padded tokens. Attention dropout (p = 0.15) prevents co-adaptation.`,
        dataFed: `Receives the bidirectional sequence tensor (batch, 200, 256) from the BiLSTM along with the binary padding mask from the tokenizer.`,
        architectureRole: `Generates a pooled 256-dimensional summary context vector for classification. Crucially, it powers the Explainable Attention Tokens displayed in the dl web UI (e.g., highlighting "writing 26%", "failure 22%", "masterpiece 18%").`,
        ablationStudy: `What happens if Multi-Head Attention is absent?
- Single Attention Degradation: Model V1 uses Bahdanau single-head additive attention, which can only focus on one dominant token at a time.
- Mean Pooling Failure: Without attention, simply averaging hidden states treats filler words ("of", "the", "an") equally with powerful sentiment keywords ("failure", "brilliant"), reducing overall F1 score by 4.2%.`,
        inputOutputSpec: {
            inputType: "torch.FloatTensor (Float32)",
            inputDimensions: "(batch_size, seq_len, 256) = (B, 200, 256)",
            outputType: "context: (B, 256), weights: (B, 4, 200, 200)",
            heads: "4 Heads, d_k = 64 each (Total = 256)",
            parameters: "3 linear projections (256x256) + 1 out projection (256x256) + biases = 263,168 params"
        },
        traversalExample: {
            "sampleText": "What an absolute masterpiece of cinematic failure. Brilliant writing!",
            "steps": [
                {
                    "stage": "Multi-Head Query, Key, Value Projections",
                    "concept": "What are Query, Key, and Value? Imagine searching a library: the Query is what you're looking for, the Key is the book's label, and the Value is the actual book content. Multi-Head Attention creates 4 parallel heads, each looking for different relationships (syntax, praise words, sarcasm clues, punctuation).",
                    "simpleInput": "BiLSTM sequence tensor (200 words x 256 features).",
                    "simpleProcessing": "Multiplies the sequence by weight matrices to create 4 sets of Queries, Keys, and Values (64 dimensions per head).",
                    "simpleOutput": "4 specialized heads ready to inspect connections across the entire review.",
                    "impact": "Enables the model to focus on subtle irony in Head 1 while tracking overall movie topic in Head 2.",
                    "math": {
                        "formula": "Q_i = H * W_Q_i, K_i = H * W_K_i, V_i = H * W_V_i, for i in [1..4]",
                        "inputTensor": "BiLSTM Context Tensor H in R^(1 x 200 x 256)",
                        "operation": "Linear projections into 4 heads: d_k = 64 per head (W_Q, W_K, W_V in R^(256 x 64)).",
                        "outputTensor": "Four Query, Key, Value tensors:\n- Head 1: Syntactic dependency and clause connection\n- Head 2: Lexical polarity extremes ('masterpiece' <-> 'failure')\n- Head 3: Sarcastic intensifier anchoring ('What an absolute', '!')\n- Head 4: Long-range affective contrast"
                    },
                    "input": "BiLSTM Context Tensor H in R^(1 x 200 x 256)",
                    "operation": "Linear projections into 4 heads: d_k = 64 per head (W_Q, W_K, W_V in R^(256 x 64)).",
                    "output": "Four Query, Key, Value tensors:\n- Head 1: Syntactic dependency and clause connection\n- Head 2: Lexical polarity extremes ('masterpiece' <-> 'failure')\n- Head 3: Sarcastic intensifier anchoring ('What an absolute', '!')\n- Head 4: Long-range affective contrast"
                },
                {
                    "stage": "Scaled Dot-Product Attention Matrices with Masking",
                    "concept": "What is Scaled Dot-Product? Computes an attention grid showing how much every word attends to every other word. Padding positions are masked out with -1e9 so the model pays zero attention to filler zeros.",
                    "simpleInput": "Query and Key matrices for all 4 heads.",
                    "simpleProcessing": "Multiplies Q by K transposed, divides by sqrt(64) = 8 to prevent exploding numbers, masks padding, and applies softmax.",
                    "simpleOutput": "Four 200x200 attention percentage maps showing which words connect with each other.",
                    "impact": "Reveals how praise words ('masterpiece') directly connect across the sentence to criticism words ('failure').",
                    "math": {
                        "formula": "Attn_Weights_i = Softmax((Q_i * K_i^T) / sqrt(64) + Mask)",
                        "inputTensor": "Q_i, K_i in R^(1 x 200 x 64) and Attention Mask (1, 200)",
                        "operation": "Dot product: A_i = (Q_i * K_i^T) / 8.0; apply -1e9 mask to positions 13-199; apply softmax along sequence axis.",
                        "outputTensor": "Attention Probability Tensor (1, 4, 200, 200) Float32:\nHead 2 allocates 41.2% attention mass between token 'failure' and token 'masterpiece', explicitly discovering cross-clause semantic contradiction."
                    },
                    "input": "Q_i, K_i in R^(1 x 200 x 64) and Attention Mask (1, 200)",
                    "operation": "Dot product: A_i = (Q_i * K_i^T) / 8.0; apply -1e9 mask to positions 13-199; apply softmax along sequence axis.",
                    "output": "Attention Probability Tensor (1, 4, 200, 200) Float32:\nHead 2 allocates 41.2% attention mass between token 'failure' and token 'masterpiece', explicitly discovering cross-clause semantic contradiction."
                },
                {
                    "stage": "Multi-Head Context Synthesis & Output Projection",
                    "concept": "What is Context Synthesis? Multiplies the attention percentages by the Value vectors to produce a concentrated summary of the most important parts of the sentence.",
                    "simpleInput": "Attention percentage grids and Value matrices across all 4 heads.",
                    "simpleProcessing": "Blends the values according to attention weights, glues the 4 heads together, and projects back to 256 dimensions.",
                    "simpleOutput": "A single 256-number context vector capturing both praise keywords and critical fault words.",
                    "impact": "Combines multi-perspective clues into a unified context representation for classification.",
                    "math": {
                        "formula": "c = LayerNorm(Dropout_0.2(concat([head_1, head_2, head_3, head_4]) * W_O) + H_summary)",
                        "inputTensor": "Attention weights and Value tensors V_i in R^(1 x 200 x 64)",
                        "operation": "Multiply weights by values, concatenate 4 heads (4 x 64 = 256), project with W_O in R^(256 x 256), add residual connection, and apply LayerNorm.",
                        "outputTensor": "Unified Context Vector c in R^256 (L2-norm: ||c||_2 = 10.15):\nDense multi-task embedding synthesizing both the sarcastic praise motif ('Brilliant writing!') and the criticism keyword ('failure')."
                    },
                    "input": "Attention weights and Value tensors V_i in R^(1 x 200 x 64)",
                    "operation": "Multiply weights by values, concatenate 4 heads (4 x 64 = 256), project with W_O in R^(256 x 256), add residual connection, and apply LayerNorm.",
                    "output": "Unified Context Vector c in R^256 (L2-norm: ||c||_2 = 10.15):\nDense multi-task embedding synthesizing both the sarcastic praise motif ('Brilliant writing!') and the criticism keyword ('failure')."
                },
                {
                    "stage": "Explainable Token Weight Extraction",
                    "concept": "What is Token Weight Extraction? Takes the attention scores and turns them into human-readable percentages so users can see exactly which words influenced the AI's decision.",
                    "simpleInput": "Attention weights averaged across all 4 heads.",
                    "simpleProcessing": "Ranks tokens by how much attention the AI paid to them.",
                    "simpleOutput": "Clear ranking: 'writing' (26%), '!' (26%), 'failure' (25%), 'Brilliant' (23%).",
                    "impact": "Provides full explainability and transparency into why the model classified the review as sarcastic.",
                    "math": {
                        "formula": "Importance_t = (1/4) * sum_i(Attn_Weights_{i, summary, t})",
                        "inputTensor": "Attention weights tensor averaged across 4 heads at summary position",
                        "operation": "Extract normalized percentage contribution per token and rank by attentive focus.",
                        "outputTensor": "Explainable Attention Token Distribution:\n- 'writing': 26% (High syntactical anchor)\n- '!': 26% (Exclamation punctuation marker)\n- 'failure': 25% (Critical fault keyword)\n- 'Brilliant': 23% (Superlative praise keyword)\nModel concentrates attention simultaneously on both the praise phrase and the contrasting fault."
                    },
                    "input": "Attention weights tensor averaged across 4 heads at summary position",
                    "operation": "Extract normalized percentage contribution per token and rank by attentive focus.",
                    "output": "Explainable Attention Token Distribution:\n- 'writing': 26% (High syntactical anchor)\n- '!': 26% (Exclamation punctuation marker)\n- 'failure': 25% (Critical fault keyword)\n- 'Brilliant': 23% (Superlative praise keyword)\nModel concentrates attention simultaneously on both the praise phrase and the contrasting fault."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 190" width="100%" height="190" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="65" width="130" height="60" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="85" y="92" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="middle">BiLSTM STATES</text>
  <text x="85" y="108" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">(B, 200, 256)</text>

  <line x1="150" y1="95" x2="210" y2="95" stroke="#4f46e5" stroke-width="2"/>

  <rect x="210" y="25" width="290" height="140" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="355" y="52" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#4338ca" text-anchor="middle">4-HEAD SCALED DOT-PRODUCT</text>
  <text x="355" y="75" font-family="'Space Mono',monospace" font-size="9" fill="#4f46e5" text-anchor="middle">Q, K, V = Linear(256 -> 64) x 4 Heads</text>
  <text x="355" y="98" font-family="'Space Mono',monospace" font-size="9" fill="#4338ca" text-anchor="middle">Weights = softmax((Q*K^T)/8 + Mask)</text>
  <text x="355" y="120" font-family="'Space Mono',monospace" font-size="9" fill="#64748b" text-anchor="middle">Context = Weights * V</text>
  <text x="355" y="142" font-family="'Space Mono',monospace" font-size="9" fill="#4338ca" text-anchor="middle">Output = Concat(heads) * W_O</text>

  <line x1="500" y1="95" x2="560" y2="95" stroke="#4f46e5" stroke-width="2"/>

  <rect x="560" y="65" width="180" height="60" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
  <text x="650" y="92" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#15803d" text-anchor="middle">ATTN CONTEXT</text>
  <text x="650" y="108" font-family="'Space Mono',monospace" font-size="8.5" fill="#166534" text-anchor="middle">(B, 256) + Token Weights</text>
</svg>`,
        relatedComponents: ["bilstm", "incongruity_module", "gated_polarity_inversion"]
    },

    gated_polarity_inversion: {
        id: "gated_polarity_inversion",
        title: "Flagship Gated Polarity Inversion Head",
        shortName: "Gated Inversion",
        category: "Non-Linear Polarity Correction",
        models: ["Model V3 (Flagship)"],
        parameters: "52,609 Dedicated Inversion & Gating Parameters",
        inputShape: "Shared Representation z in R^256 + Sarcasm Probability p_sarc in [0, 1]",
        outputShape: "Inverted Features (B, 256) -> Conditioned Sentiment Logits (B, 1)",
        badge: "Flagship V3 Core Breakthrough",
        summary: "Multiplicative gating head that dynamically inverts surface positive representations into negative polarity when sarcasm is detected, resolving sarcastic criticism and mock praise.",
        beginnerIntuition: "Think of Gated Inversion like a polarity inverter switch on an electrical circuit. When the Sarcasm detector stays at 0 (literal praise), the switch stays at +1.0 (positive). But the moment the sarcasm alarm trips to 1.0 (sarcasm detected!), the gate flips to -1.0, multiplying the praise by -1 and converting 'Brilliant writing!' into its true intended meaning: scathing criticism!",
        whatIsIt: `The Gated Polarity Inversion Head is the primary architectural innovation of dl V3. It solves the famous 'Mock Praise Dilemma' in deep sentiment analysis: when a user expresses criticism through superficial praise ("Brilliant writing! A 20-minute dance sequence in a psychological thriller. Masterpiece."), standard neural networks fail because positive words flood the feature space.
V3 introduces a continuous multiplicative gate gamma:
1. Sarcasm Probability: p_sarc = sigmoid(W_sarc * z + b_sarc) in [0.0, 1.0]
2. Inversion Multiplier: gamma = 1.0 - 2.0 * p_sarc in [-1.0, +1.0]
3. Feature Inversion: z_inverted = z * gamma
4. Sentiment Input: [z ; p_sarc ; z_inverted] -> Linear -> Sentiment Logits
When p_sarc ≈ 0 (literal), gamma ≈ +1.0, preserving natural sentiment.
When p_sarc ≈ 1 (sarcastic), gamma ≈ -1.0, mathematically flipping positive activations into negative territory!`,
        howIsItTrained: `Trained jointly with a dual loss function:
L_total = 0.85 * BCE(y_sent_hat, y_sent) + 0.15 * BCE(y_sarc_hat, y_sarc)
During backpropagation, errors in inverted mock praise sentences backpropagate directly through gamma into the sarcasm head, creating a strong synergistic gradient signal that forces the sarcasm head to sharpen its boundary.`,
        dataFed: `Trained on the 230K hybrid corpus with targeted hard-negative sarcasm mining from SARC Reddit and iSarcasm.`,
        architectureRole: `Serves as the decision-making brain in Model V3. When sarcasm is active, the UI displays the green 'POLARITY INVERSION ACTIVE' badge, and the interpretation correctly reads 'Sarcastic Criticism (Mock Praise)' rather than 'Positive'.`,
        ablationStudy: `What happens if Gated Polarity Inversion is absent?
- The Mock Praise Failure: Without the inversion gate, Model V2 (which has multi-task heads but no inversion) detects sarcasm (p_sarc = 0.94) but still outputs 'Positive Sentiment' (59% confidence) because the positive words 'masterpiece' and 'brilliant' overpower the linear classifier.
- In our test benchmark of 100 hard mock praise sentences, Model V2 scored only 68% accuracy, whereas Model V3 achieved 100% Mock Praise Accuracy.`,
        inputOutputSpec: {
            inputType: "torch.FloatTensor (Float32)",
            inputDimensions: "shared_repr: (B, 256), sarcasm_prob: (B, 1)",
            outputType: "torch.FloatTensor (Float32)",
            outputDimensions: "sentiment_prob: (B,), sarcasm_prob: (B,)",
            multiplierFormula: "gamma = 1.0 - 2.0 * sarcasm_prob",
            parameters: "Inversion projection + sentiment head = 52,609 parameters"
        },
        traversalExample: {
            "sampleText": "Brilliant writing! An unprompted 20-minute dance sequence. Masterpiece.",
            "steps": [
                {
                    "stage": "Shared Latent Representation z",
                    "concept": "What is Latent Vector z? A latent vector is a compact 256-number digital summary fingerprint of the entire review. Instead of juggling 200 separate words, the network condenses all the grammar, emotions, topic motifs, and contradiction alerts into a single unified vector z. It serves as the model's unified mental image before making predictions.",
                    "simpleInput": "Attention context vector (which words matter most) combined with incongruity state (where contradiction clashed).",
                    "simpleProcessing": "Passes the fused information through a dense neural projection with Mish activation to construct the 256-dimensional vector z.",
                    "simpleOutput": "A 256-number vector z where positive coordinates mark praise ('masterpiece', 'brilliant') and negative coordinates mark faults.",
                    "impact": "Serves as the single unified source of truth for both the sarcasm detector and the sentiment classifier.",
                    "math": {
                        "formula": "z = Mish(W_shared * [c ; inc] + b_shared)",
                        "inputTensor": "c in R^256 (Attention Context) and inc in R^128 (Incongruity State)",
                        "operation": "Dense linear projection from R^384 to R^256 followed by Mish non-linearity.",
                        "outputTensor": "Shared Latent Vector z in R^256 (L2-norm: ||z||_2 = 12.34):\nPositive aesthetic features show large positive magnitudes: dim 18 ('masterpiece') = +2.45, dim 42 ('brilliant') = +2.18, dim 74 ('!') = +1.98."
                    },
                    "input": "Combined Context + Incongruity feature vectors",
                    "operation": "Project through shared dense projection layer: z = Mish(W_shared * [c ; inc] + b).",
                    "output": "Shared Latent Vector z in R^256 (L2-norm: ||z||_2 = 12.34):\nPositive aesthetic features show large positive magnitudes: dim 18 ('masterpiece') = +2.45, dim 42 ('brilliant') = +2.18, dim 74 ('!') = +1.98."
                },
                {
                    "stage": "Auxiliary Sarcasm Probability Evaluation",
                    "concept": "What is Sarcasm Evaluation? The auxiliary sarcasm head examines latent vector z for telltale signs of irony - such as ecstatic praise paired with an incongruity spike.",
                    "simpleInput": "The 256-number latent vector z.",
                    "simpleProcessing": "Multiplies z by learned sarcasm detection weights and squashes the result into a probability between 0% and 100%.",
                    "simpleOutput": "94.0% Sarcasm Probability (0.940). The model is extremely confident this review is sarcastic mock-praise.",
                    "impact": "Alerts the network that surface praise is deceptive, preparing the gate to invert the sentiment.",
                    "math": {
                        "formula": "p_sarc = sigmoid(W_sarc * z + b_sarc) = 1 / (1 + exp(-(W_sarc * z + b_sarc)))",
                        "inputTensor": "Shared latent vector z in R^256",
                        "operation": "Compute logit: logit_sarc = W_sarc * z + b; squash with sigmoid: p_sarc = 1 / (1 + e^(-logit_sarc)).",
                        "outputTensor": "Sarcasm Output: Logit = +2.75 -> Sarcasm Probability p_sarc = 0.940 (94.0% confidence Sarcastic / Mock Praise, alerted by sequence incongruity 1.29)."
                    },
                    "input": "Shared latent vector z in R^256",
                    "operation": "Compute logit: logit_sarc = W_sarc * z + b; squash with sigmoid: p_sarc = 1 / (1 + e^(-logit_sarc)).",
                    "output": "Sarcasm Output: Logit = +2.75 -> Sarcasm Probability p_sarc = 0.940 (94.0% confidence Sarcastic / Mock Praise, alerted by sequence incongruity 1.29)."
                },
                {
                    "stage": "Linear Inversion Multiplier Computation",
                    "concept": "What is the Inversion Multiplier gamma? A mathematical toggle switch. If a review is sincere (0% sarcasm), gamma is positive (+1.0), keeping praise positive. If a review is sarcastic (100% sarcasm), gamma flips negative (-1.0), reversing fake praise into real criticism.",
                    "simpleInput": "Sarcasm probability p_sarc = 0.940.",
                    "simpleProcessing": "Calculates gamma = 1.0 - 2.0 * p_sarc. With p_sarc = 0.94, gamma = 1.0 - 1.88 = -0.880.",
                    "simpleOutput": "A negative inversion multiplier gamma = -0.880.",
                    "impact": "Provides the mathematical signal to reverse the emotional direction of praise features in tensor space.",
                    "math": {
                        "formula": "gamma = 1.0 - 2.0 * p_sarc",
                        "inputTensor": "Continuous sarcasm probability p_sarc = 0.940",
                        "operation": "Compute gating multiplier: gamma = 1.0 - 2.0 * p_sarc.",
                        "outputTensor": "Inversion Multiplier gamma = 1.0 - 2.0 * (0.940) = -0.880:\n- Literal statements (p_sarc ~ 0.05) yield gamma ~ +0.90 (Positive reinforcement)\n- Sarcastic statements (p_sarc = 0.94) yield negative multiplier gamma = -0.88 (Active polarity inversion)"
                    },
                    "input": "Continuous sarcasm probability p_sarc = 0.940",
                    "operation": "Compute gating multiplier: gamma = 1.0 - 2.0 * p_sarc.",
                    "output": "Inversion Multiplier gamma = 1.0 - 2.0 * (0.940) = -0.880:\n- Literal statements (p_sarc ~ 0.05) yield gamma ~ +0.90 (Positive reinforcement)\n- Sarcastic statements (p_sarc = 0.94) yield negative multiplier gamma = -0.88 (Active polarity inversion)"
                },
                {
                    "stage": "Multiplicative Inversion Gate",
                    "concept": "What is the Inversion Gate? This is where sarcasm is resolved: positive praise features ('masterpiece', 'brilliant') are multiplied by the negative multiplier gamma (-0.880), flipping their sign from positive to negative in tensor space.",
                    "simpleInput": "The positive praise coordinates in z and the negative multiplier gamma = -0.880.",
                    "simpleProcessing": "Multiplies praise features by -0.880 while preserving neutral context coordinates.",
                    "simpleOutput": "An inverted vector z_inverted where 'masterpiece' (+2.45) becomes -2.156 and 'brilliant' (+2.18) becomes -1.918.",
                    "impact": "Directly flips fake positive praise into genuine negative criticism before the final sentiment decision.",
                    "math": {
                        "formula": "z_inverted = gamma * z_positive_subspace + (1 - |gamma|) * z_neutral",
                        "inputTensor": "Positive feature subspace of z and negative multiplier gamma = -0.880",
                        "operation": "Compute modulated feature vector: z_inverted = gamma * z_positive_subspace + (1 - |gamma|) * z_neutral.",
                        "outputTensor": "Inverted Latent Vector z_inverted in R^256:\n- Dim 18 ('masterpiece'): +2.45 * (-0.880) = -2.156 (Inverted to negative sentiment)\n- Dim 42 ('brilliant'): +2.18 * (-0.880) = -1.918 (Inverted to negative sentiment)\nSurface praise is mathematically flipped into negative semantic criticism in tensor space."
                    },
                    "input": "Positive feature subspace of z and negative multiplier gamma = -0.880",
                    "operation": "Compute modulated feature vector: z_inverted = gamma * z_positive_subspace + (1 - |gamma|) * z_neutral.",
                    "output": "Inverted Latent Vector z_inverted in R^256:\n- Dim 18 ('masterpiece'): +2.45 * (-0.880) = -2.156 (Inverted to negative sentiment)\n- Dim 42 ('brilliant'): +2.18 * (-0.880) = -1.918 (Inverted to negative sentiment)\nSurface praise is mathematically flipped into negative semantic criticism in tensor space."
                },
                {
                    "stage": "Final Calibrated Sentiment Classification",
                    "concept": "What is Final Classification? The model combines the original summary, the sarcasm probability, and the inverted criticism vector to make the final true sentiment prediction.",
                    "simpleInput": "Original vector z + Sarcasm probability 94% + Inverted vector z_inverted.",
                    "simpleProcessing": "Projects the combined 513-dimensional evidence into a sentiment logit and squashes to [0, 1].",
                    "simpleOutput": "8.0% Positive, 92.0% Negative -> Final Verdict: Scathingly NEGATIVE.",
                    "impact": "Correctly decodes sarcastic mock praise as scathing negative criticism, eliminating the classic NLP sarcasm blindspot.",
                    "math": {
                        "formula": "p_pos = sigmoid(W_sent * [z ; p_sarc ; z_inverted] + b_sent)",
                        "inputTensor": "Concatenated vector [z ; p_sarc ; z_inverted] of dimension 513",
                        "operation": "Compute final logit: logit_sent = W_sent * [z, p_sarc, z_inverted] + b; apply sigmoid: p_pos = sigmoid(logit_sent).",
                        "outputTensor": "Final Sentiment Output: Calibrated Logit = -2.44 -> Positive Probability = 0.080 (Negative Probability = 92.0%).\nFinal Classification: NEGATIVE (Correctly decodes sarcastic mock praise as scathing negative criticism)."
                    },
                    "input": "Concatenated vector [z ; p_sarc ; z_inverted] of dimension 513",
                    "operation": "Compute final logit: logit_sent = W_sent * [z, p_sarc, z_inverted] + b; apply sigmoid: p_pos = sigmoid(logit_sent).",
                    "output": "Final Sentiment Output: Calibrated Logit = -2.44 -> Positive Probability = 0.080 (Negative Probability = 92.0%).\nFinal Classification: NEGATIVE (Correctly decodes sarcastic mock praise as scathing negative criticism)."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 210" width="100%" height="210" xmlns="http://www.w3.org/2000/svg">
  <rect x="15" y="75" width="140" height="60" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="85" y="102" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="middle">SHARED REPR z</text>
  <text x="85" y="118" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">(B, 256) Features</text>

  <path d="M 155 105 L 200 45 L 230 45" fill="none" stroke="#4f46e5" stroke-width="1.8"/>
  <path d="M 155 105 L 200 160 L 230 160" fill="none" stroke="#4f46e5" stroke-width="1.8"/>

  <rect x="230" y="20" width="200" height="50" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="330" y="44" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#b45309" text-anchor="middle">SARCASM HEAD</text>
  <text x="330" y="60" font-family="'Space Mono',monospace" font-size="8.5" fill="#92400e" text-anchor="middle">p_sarc = sigmoid(W*z + b) = 0.94</text>

  <rect x="230" y="135" width="200" height="50" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="330" y="158" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#4338ca" text-anchor="middle">INVERSION GATE</text>
  <text x="330" y="174" font-family="'Space Mono',monospace" font-size="8.5" fill="#6366f1" text-anchor="middle">gamma = 1 - 2*p_sarc = -0.88</text>

  <path d="M 430 45 L 470 45 L 490 85" fill="none" stroke="#f59e0b" stroke-width="1.8"/>
  <path d="M 430 160 L 470 160 L 490 125" fill="none" stroke="#6366f1" stroke-width="1.8"/>

  <rect x="490" y="65" width="250" height="80" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
  <text x="615" y="94" font-family="'Space Grotesk',sans-serif" font-size="11" font-weight="700" fill="#15803d" text-anchor="middle">SENTIMENT CLASSIFIER</text>
  <text x="615" y="112" font-family="'Space Mono',monospace" font-size="9" fill="#166534" text-anchor="middle">Input: [z ; p_sarc ; z * (-0.88)]</text>
  <text x="615" y="130" font-family="'Space Grotesk',sans-serif" font-size="10" font-weight="700" fill="#15803d" text-anchor="middle">RESULT: NEGATIVE (Mock Praise Resolved)</text>
</svg>`,
        relatedComponents: ["incongruity_module", "multihead_attention", "multitask_heads"]
    },

    multitask_heads: {
        id: "multitask_heads",
        title: "Joint Dual-Task Classification Heads",
        shortName: "Dual Classification Heads",
        category: "Multi-Task Learning & Joint Loss Optimization",
        models: ["Model V2 (Multi-Task)", "Model V3 (Flagship)"],
        parameters: "66,306 Parameters (Sentiment Projection + Sarcasm Projection)",
        inputShape: "Shared latent representation (batch_size, 256)",
        outputShape: "Sentiment Probability (B,) + Sarcasm Probability (B,)",
        badge: "Joint Multi-Task Architecture",
        summary: "Dual classification projections trained simultaneously on complementary objectives: binary Sentiment (Positive/Negative) and binary Sarcasm (Sarcastic/Literal).",
        beginnerIntuition: "Think of this like teaching a student two related subjects simultaneously (e.g., Geometry and Physics). By learning to predict both genuine sentiment and sarcasm at the same time, the student develops a much deeper understanding than if they were only trained on one subject alone.",
        whatIsIt: `Multi-Task Learning (MTL) is a paradigm in machine learning where multiple related tasks share common lower-level representations. In dl V2 and V3, rather than training a sentiment classifier in isolation, the architecture trains two heads in parallel:
1. Sentiment Head: Predicts binary sentiment (0 = Negative, 1 = Positive)
2. Sarcasm Head: Predicts binary sarcasm (0 = Literal, 1 = Sarcastic)
The shared representations are forced to capture both emotional valence and ironic intent simultaneously.`,
        howIsItTrained: `Optimized with a weighted multi-task objective function:
L_total = alpha * L_BCE(y_sent_hat, y_sent) + beta * L_BCE(y_sarc_hat, y_sarc)
where alpha = 0.85 (primary sentiment goal) and beta = 0.15 (auxiliary sarcasm regularization). The auxiliary sarcasm gradient acts as an inductive bias, preventing the shared CNN and BiLSTM from collapsing into a naive bag-of-words keyword counter.`,
        dataFed: `Trained on multi-labeled hybrid corpora with both sentiment polarity ground truth and sarcasm annotation flags.`,
        architectureRole: `Outputs the dual probabilities seen in the dl UI: Sentiment (Positive/Negative with confidence percentage) and Sarcasm (Literal/Sarcastic with confidence percentage).`,
        ablationStudy: `What happens if Multi-Task Learning is absent?
- Single-Task Blindness: In Model V1, only a sentiment head exists. When confronted with sarcasm, it has zero capacity to flag irony, systematically failing on sarcastic praise.
- Multi-Task Regularization: In empirical tests, training sentiment jointly with sarcasm improved clean sentiment F1 score by +1.6% due to shared representation regularization.`,
        inputOutputSpec: {
            inputType: "torch.FloatTensor (Float32) shape (batch_size, 256)",
            outputType: "Dict with sentiment_prob (B,) and sarcasm_prob (B,)",
            parameters: "Dense projections + biases = 66,306 params"
        },
        traversalExample: {
            "sampleText": "Dual task execution on sarcastic review",
            "steps": [
                {
                    "stage": "Shared Latent Feature Bifurcation",
                    "concept": "What is Latent Feature Bifurcation? Latent vector z is a 256-number summary fingerprint of the review. Bifurcation means splitting this single vector into two parallel pipelines so sentiment and sarcasm can be analyzed at the exact same time without duplicating work.",
                    "simpleInput": "Latent feature vector z in R^256 (a compact 256-number summary of words, emotions, and tone).",
                    "simpleProcessing": "Duplicates the single vector z into two parallel objective streams: Branch A (evaluating sentiment) and Branch B (evaluating sarcasm).",
                    "simpleOutput": "Two identical 256-number vectors fed into parallel task heads.",
                    "impact": "Allows both heads to share knowledge; learning what sarcasm looks like actively helps the network judge sentiment accurately.",
                    "math": {
                        "formula": "z_sent = z, z_sarc = z (Shared branch broadcast)",
                        "inputTensor": "Latent feature vector z in R^256",
                        "operation": "Split shared latent representation into two parallel objective pathways: Branch A (Sentiment) and Branch B (Sarcasm).",
                        "outputTensor": "Two Parallel Task Vectors: Branch A in R^256 Float32 + Branch B in R^256 Float32.\nBoth tasks evaluate the identical underlying representation, enforcing joint learning of affective syntax and irony markers."
                    },
                    "input": "Latent feature vector z in R^256",
                    "operation": "Split shared latent representation into two parallel objective pathways: Branch A (Sentiment) and Branch B (Sarcasm).",
                    "output": "Two Parallel Task Vectors: Branch A in R^256 Float32 + Branch B in R^256 Float32.\nBoth tasks evaluate the identical underlying representation, enforcing joint learning of affective syntax and irony markers."
                },
                {
                    "stage": "Independent Linear Projections (Logits)",
                    "concept": "What are Logits? Logits are raw uncalibrated scores showing how strongly the model leans toward positive sentiment or sarcastic tone before converting to percentages.",
                    "simpleInput": "Branch A vector and Branch B vector.",
                    "simpleProcessing": "Multiplies each branch by task-specific weight matrices to calculate raw unbounded confidence scores.",
                    "simpleOutput": "Sentiment Logit: -1.45 (leans negative) | Sarcasm Logit: +1.62 (strongly leans sarcastic).",
                    "impact": "Separates emotional tone (positive vs negative) from irony detection (literal vs sarcastic).",
                    "math": {
                        "formula": "logit_sent = W_sent * z_A + b_sent | logit_sarc = W_sarc * z_B + b_sarc",
                        "inputTensor": "Branch A vector in R^256 and Branch B vector in R^256",
                        "operation": "Compute raw logits: logit_sent = W_sent * z_A + b_sent | logit_sarc = W_sarc * z_B + b_sarc.",
                        "outputTensor": "Raw Unbounded Logits:\n- Sentiment Logit = -1.45 (Distance to decision boundary = 1.45 in negative half-space)\n- Sarcasm Logit = +1.62 (Distance to decision boundary = 1.62 in sarcastic half-space)"
                    },
                    "input": "Branch A vector in R^256 and Branch B vector in R^256",
                    "operation": "Compute raw logits: logit_sent = W_sent * z_A + b_sent | logit_sarc = W_sarc * z_B + b_sarc.",
                    "output": "Raw Unbounded Logits:\n- Sentiment Logit = -1.45 (Distance to decision boundary = 1.45 in negative half-space)\n- Sarcasm Logit = +1.62 (Distance to decision boundary = 1.62 in sarcastic half-space)"
                },
                {
                    "stage": "Sigmoidal Probability Calibration",
                    "concept": "What is Sigmoid Squashing? Raw logits can be any positive or negative number. Sigmoid converts them smoothly into easy-to-read probabilities between 0% and 100%.",
                    "simpleInput": "Raw logits: Sentiment (-1.45) and Sarcasm (+1.62).",
                    "simpleProcessing": "Passes both logits through the standard S-curve function: 1 / (1 + e^-x).",
                    "simpleOutput": "Sentiment: 81.0% Negative (19.0% Positive) | Sarcasm: 83.5% Sarcastic (16.5% Literal).",
                    "impact": "Delivers human-readable percentages with calibrated confidence for both tasks in a single forward pass.",
                    "math": {
                        "formula": "p = 1 / (1 + exp(-logit))",
                        "inputTensor": "Raw logits: logit_sent = -1.45, logit_sarc = +1.62",
                        "operation": "Apply logistic sigmoid: p = 1 / (1 + e^(-logit)) for both objective outputs.",
                        "outputTensor": "Dual Calibrated Output Probabilities:\n- Sentiment Head: p_pos = 0.190 -> Negative: 81.0%, Positive: 19.0%\n- Sarcasm Head: p_sarc = 0.835 -> Sarcastic: 83.5%, Literal: 16.5%\nSimultaneous calibrated predictions generated from a single forward pass."
                    },
                    "input": "Raw logits: logit_sent = -1.45, logit_sarc = +1.62",
                    "operation": "Apply logistic sigmoid: p = 1 / (1 + e^(-logit)) for both objective outputs.",
                    "output": "Dual Calibrated Output Probabilities:\n- Sentiment Head: p_pos = 0.190 -> Negative: 81.0%, Positive: 19.0%\n- Sarcasm Head: p_sarc = 0.835 -> Sarcastic: 83.5%, Literal: 16.5%\nSimultaneous calibrated predictions generated from a single forward pass."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 170" width="100%" height="170" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="55" width="150" height="60" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="95" y="82" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="middle">SHARED REPR z</text>
  <text x="95" y="98" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">(B, 256) Dimensions</text>

  <path d="M 170 85 L 240 45 L 280 45" fill="none" stroke="#4f46e5" stroke-width="1.8"/>
  <path d="M 170 85 L 240 125 L 280 125" fill="none" stroke="#4f46e5" stroke-width="1.8"/>

  <rect x="280" y="20" width="220" height="50" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
  <text x="390" y="44" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#15803d" text-anchor="middle">SENTIMENT HEAD (alpha=0.85)</text>
  <text x="390" y="60" font-family="'Space Mono',monospace" font-size="8.5" fill="#166534" text-anchor="middle">Positive vs Negative (Loss: BCE)</text>

  <rect x="280" y="100" width="220" height="50" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>
  <text x="390" y="124" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#b45309" text-anchor="middle">SARCASM HEAD (beta=0.15)</text>
  <text x="390" y="140" font-family="'Space Mono',monospace" font-size="8.5" fill="#92400e" text-anchor="middle">Sarcastic vs Literal (Loss: BCE)</text>

  <line x1="500" y1="45" x2="560" y2="45" stroke="#22c55e" stroke-width="1.8"/>
  <line x1="500" y1="125" x2="560" y2="125" stroke="#f59e0b" stroke-width="1.8"/>

  <rect x="560" y="20" width="180" height="50" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="650" y="44" font-family="'Space Grotesk',sans-serif" font-size="10" font-weight="700" fill="#0f172a" text-anchor="middle">p_sent in [0, 1]</text>
  <text x="650" y="60" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">Sentiment Probability</text>

  <rect x="560" y="100" width="180" height="50" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>
  <text x="650" y="124" font-family="'Space Grotesk',sans-serif" font-size="10" font-weight="700" fill="#0f172a" text-anchor="middle">p_sarc in [0, 1]</text>
  <text x="650" y="140" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">Sarcasm Probability</text>
</svg>`,
        relatedComponents: ["gated_polarity_inversion", "incongruity_module"]
    },

    whitespace_tokenizer: {
        id: "whitespace_tokenizer",
        title: "Whitespace & Punctuation Word Tokenizer",
        shortName: "Whitespace Tokenizer",
        category: "Baseline Preprocessing",
        models: ["Model V1 (Baseline)"],
        parameters: "35,000 Word Dictionary (0 Trainable Weights)",
        inputShape: "Raw Text String (UTF-8)",
        outputShape: "Tensor (batch_size, 200) Int64 IDs",
        badge: "V1 Legacy Baseline",
        summary: "Traditional word-level tokenizer using regular expression splitting on spaces and punctuation, with a fixed vocabulary of the top 35,000 most frequent tokens.",
        beginnerIntuition: "Think of Whitespace Tokenization like a rigid dictionary with a padlocked cover. If a word isn't listed under its exact spelling—such as internet slang, a typo, or a hyphenated word—the reader is forced to throw their hands up and declare it 'Unknown' ([UNK]), completely blinding the model.",
        whatIsIt: `The Whitespace Word Tokenizer is the baseline preprocessor used in Model V1. It lowercases input text, separates punctuation characters from words, and splits the string strictly by whitespace. Words are mapped to numerical IDs based on an index built from the top 35,000 most frequent whole words in the IMDB training corpus. Any word outside this fixed vocabulary is permanently mapped to [UNK] (ID 1).`,
        howIsItTrained: `Constructed by scanning the IMDB 50,000 review dataset, compiling a frequency histogram of unique words, and retaining the top 35,000 whole words. No subword merging is performed.`,
        dataFed: `Trained exclusively on formal IMDB movie reviews.`,
        architectureRole: `Feeds Model V1's random 128d trainable embedding layer.`,
        ablationStudy: `Why it was superseded in V2 and V3:
- Rigid Whole-Word Matching: Cannot handle compound words, typos, or slang.
- Out-Of-Vocabulary Blindness: In conversational datasets, up to 15% of words are mapped to [UNK], stripping the model of crucial nuances.`,
        inputOutputSpec: {
            inputType: "Raw text string",
            outputType: "torch.LongTensor of shape (B, 200)",
            specialTokens: "[PAD] = 0, [UNK] = 1"
        },
        traversalExample: {
            "sampleText": "What a cinematic masterpiece!",
            "steps": [
                {
                    "stage": "Lowercasing & Punctuation Splitting",
                    "concept": "What is Whitespace Tokenization? The simplest text splitting method: converts text to lowercase and cuts strictly wherever there is a space. Unlike BPE, it cannot break words down into smaller subword morphemes.",
                    "simpleInput": "Raw text string: 'What a cinematic masterpiece!'",
                    "simpleProcessing": "Converts all letters to lowercase and splits strictly on space boundaries.",
                    "simpleOutput": "5 whole words: ['what', 'a', 'cinematic', 'masterpiece', '!'].",
                    "impact": "Any rare, misspelled, or compound word becomes a separate unknown entity, crippling vocabulary coverage.",
                    "math": {
                        "formula": "Tokens = Split(Lower(T_raw), whitespace)",
                        "inputTensor": "Raw string (30 chars)",
                        "operation": "Convert to lowercase, isolate punctuation characters [. , ! ?], and split strictly on whitespace.",
                        "outputTensor": "5 word tokens (Array length: 5):\n['what', 'a', 'cinematic', 'masterpiece', '!']\n(Compound or unseen rare words cannot be broken down sub-lexically)"
                    },
                    "input": "What a cinematic masterpiece!",
                    "operation": "Convert to lowercase, isolate punctuation characters [. , ! ?], and split strictly on whitespace.",
                    "output": "5 word tokens (Array length: 5):\n['what', 'a', 'cinematic', 'masterpiece', '!']\n(Compound or unseen rare words cannot be broken down sub-lexically)"
                },
                {
                    "stage": "Vocabulary Dictionary Lookup & Zero-Padding",
                    "concept": "What is Out-Of-Vocabulary (OOV) Mapping? The tokenizer looks up each word in a fixed list. If a word wasn't in the list during training, it is permanently replaced with [OOV] ID 1, losing all meaning.",
                    "simpleInput": "The 5 word tokens.",
                    "simpleProcessing": "Replaces known words with IDs, maps unknown words to ID 1 ([OOV]), and pads with zeros to length 200.",
                    "simpleOutput": "A 200-number list with 5 valid IDs and 195 padding zeros.",
                    "impact": "Any word not seen in the 35,000 dictionary is permanently erased, blinding Model V1 to novel slang or compound words.",
                    "math": {
                        "formula": "ID_t = Vocab[w_t] if w_t in Vocab else 1",
                        "inputTensor": "5 word tokens",
                        "operation": "Lookup each word in the 35,000 word-level dictionary; map unknown words to ID 1 ([OOV]); right-pad with ID 0 up to 200.",
                        "outputTensor": "Int64 Tensor (batch_size=1, seq_len=200):\n[14, 3, 2812, 5940, 4, 0, 0, 0, ... 0]\n(5 valid word indices + 195 zero-padding units. Any word not in 35K list irrevocably maps to [OOV] ID 1, losing all lexical semantics)"
                    },
                    "input": "5 word tokens",
                    "operation": "Lookup each word in the 35,000 word-level dictionary; map unknown words to ID 1 ([OOV]); right-pad with ID 0 up to 200.",
                    "output": "Int64 Tensor (batch_size=1, seq_len=200):\n[14, 3, 2812, 5940, 4, 0, 0, 0, ... 0]\n(5 valid word indices + 195 zero-padding units. Any word not in 35K list irrevocably maps to [OOV] ID 1, losing all lexical semantics)"
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 140" width="100%" height="140" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="35" width="180" height="60" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="110" y="62" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="middle">RAW TEXT</text>
  <text x="110" y="78" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">"What a masterpiece"</text>

  <line x1="200" y1="65" x2="270" y2="65" stroke="#4f46e5" stroke-width="2"/>

  <rect x="270" y="30" width="220" height="70" rx="8" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5"/>
  <text x="380" y="58" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#b91c1c" text-anchor="middle">WHITESPACE REGEX</text>
  <text x="380" y="74" font-family="'Space Mono',monospace" font-size="8.5" fill="#dc2626" text-anchor="middle">Fixed 35K Dictionary</text>
  <text x="380" y="90" font-family="'Space Mono',monospace" font-size="8.5" fill="#ef4444" text-anchor="middle">Unseen words -> [UNK]</text>

  <line x1="490" y1="65" x2="560" y2="65" stroke="#4f46e5" stroke-width="2"/>

  <rect x="560" y="35" width="180" height="60" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="650" y="62" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="middle">TOKEN IDS</text>
  <text x="650" y="78" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">(B, 200) Int64 IDs</text>
</svg>`,
        relatedComponents: ["bpe_tokenizer", "trainable_embeddings"]
    },

    trainable_embeddings: {
        id: "trainable_embeddings",
        title: "Randomly Initialized 128d Trainable Embeddings",
        shortName: "Trainable Embeddings",
        category: "Baseline Embedding Layer",
        models: ["Model V1 (Baseline)"],
        parameters: "4,480,000 Parameters (35,000 Vocab x 128 Dim)",
        inputShape: "Tensor (batch_size, 200) Int64 Token IDs",
        outputShape: "Tensor (batch_size, 200, 128) Float32 Embeddings",
        badge: "V1 Random Weights",
        summary: "128-dimensional embedding matrix initialized with uniform random noise U(-0.05, +0.05) and trained from scratch.",
        beginnerIntuition: "Think of trainable embeddings from scratch like a student trying to learn the entire English language from reading only movie reviews without a dictionary. Without prior knowledge of what words mean, it takes immense data to learn that 'good' and 'great' are related.",
        whatIsIt: "Model V1's baseline embedding layer. Words are assigned arbitrary random vectors at initialization and must learn semantic relationships purely through backpropagation on sentiment loss.",
        howIsItTrained: "Trained using Adam optimizer from uniform random weights without pre-trained language corpora.",
        dataFed: "IMDB 50K reviews.",
        architectureRole: "Maps token IDs to 128d space for Conv1D.",
        ablationStudy: "Suffers from severe overfitting and slow convergence compared to GloVe 300d (+3.5x slower).",
        inputOutputSpec: { inputType: "LongTensor (B, 200)", outputType: "FloatTensor (B, 200, 128)" },
        traversalExample: {
            "sampleText": "Word ID 2812 ('cinematic')",
            "steps": [
                {
                    "stage": "Random Gaussian Weight Table Initialization",
                    "concept": "What are Trainable Random Embeddings? Instead of using pre-trained word vectors like GloVe, Model V1 starts with a table filled with completely random numbers. The model has zero knowledge of English grammar or word meanings.",
                    "simpleInput": "Dictionary size of 35,000 words and embedding dimension 128.",
                    "simpleProcessing": "Generates 4.48 million random numbers drawn from a Gaussian normal distribution.",
                    "simpleOutput": "A weight table where every word starts as meaningless random static.",
                    "impact": "Forces the network to learn English from scratch on a small dataset, leading to severe overfitting and poor generalization.",
                    "math": {
                        "formula": "E ~ Normal(0, 1 / sqrt(128))",
                        "inputTensor": "Vocabulary size = 35,000, embedding dimension = 128",
                        "operation": "Initialize weight matrix E ~ N(0, 1 / sqrt(128)) with uniform variance 0.0078.",
                        "outputTensor": "Trainable Parameter Tensor E of shape (35000, 128) Float32 (Total parameters: 4,480,000):\nAll 35,000 word vectors begin as uninformative Gaussian random noise with zero semantic clustering."
                    },
                    "input": "Vocabulary size = 35,000, embedding dimension = 128",
                    "operation": "Initialize weight matrix E ~ N(0, 1 / sqrt(128)) with uniform variance 0.0078.",
                    "output": "Trainable Parameter Tensor E of shape (35000, 128) Float32 (Total parameters: 4,480,000):\nAll 35,000 word vectors begin as uninformative Gaussian random noise with zero semantic clustering."
                },
                {
                    "stage": "Token Index Gathering & Gradient Backpropagation",
                    "concept": "What is Embedding Gathering & Updating? As training reviews pass through, the model updates word numbers using gradient descent. Words that appear frequently in training get tuned, but rare words stay stuck as random noise.",
                    "simpleInput": "Token ID 2812 ('cinematic').",
                    "simpleProcessing": "Pulls out row 2812 and nudges its 128 numbers during each training step.",
                    "simpleOutput": "A 128-number vector shaped purely by the movie review training set.",
                    "impact": "Fails to generalize on unseen words, leaving Model V1 vulnerable to unfamiliar movie reviews.",
                    "math": {
                        "formula": "v = E[id]; E = E - lr * (dL / dE)",
                        "inputTensor": "Token ID 2812 ('cinematic')",
                        "operation": "Extract row 2812: v = E[2812]; update coordinates during backward pass via Adam optimizer: E = E - lr * dL / dE.",
                        "outputTensor": "Dense Float32 Vector in R^128 (L2-norm: ||v||_2 = 3.65):\nCoordinates shaped exclusively by the 250K sentiment training reviews. Words not frequent in training receive few gradient updates and stay close to initial random noise."
                    },
                    "input": "Token ID 2812 ('cinematic')",
                    "operation": "Extract row 2812: v = E[2812]; update coordinates during backward pass via Adam optimizer: E = E - lr * dL / dE.",
                    "output": "Dense Float32 Vector in R^128 (L2-norm: ||v||_2 = 3.65):\nCoordinates shaped exclusively by the 250K sentiment training reviews. Words not frequent in training receive few gradient updates and stay close to initial random noise."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg"><rect x="30" y="30" width="700" height="60" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/><text x="380" y="65" font-family="'Space Grotesk',sans-serif" font-size="12" font-weight="700" fill="#0f172a" text-anchor="middle">RANDOM UNIFORM LOOKUP (35,000 x 128)</text></svg>`,
        relatedComponents: ["glove_embeddings", "whitespace_tokenizer"]
    },

    conv1d_v1: {
        id: "conv1d_v1",
        title: "3-Scale Parallel 1D Convolutions with ReLU",
        shortName: "Conv1D Filters",
        category: "Baseline Feature Extraction",
        models: ["Model V1 (Baseline)"],
        parameters: "61,536 Parameters (k=3,5,7 with 32 filters each)",
        inputShape: "Tensor (batch_size, 128, 200)",
        outputShape: "Tensor (batch_size, 96, 200)",
        badge: "V1 Standard CNN",
        summary: "Baseline convolutional filter bank capturing 3-gram, 5-gram, and 7-gram phrases with standard ReLU activation.",
        beginnerIntuition: "Traditional 1D convolutions scan text with a single set of filters, missing the multi-scale nuance captured by modern parallel Inception blocks.",
        whatIsIt: "V1 baseline feature extractor using 3 parallel kernels (3, 5, 7) with 32 filters each.",
        howIsItTrained: "Trained end-to-end with standard ReLU activation.",
        dataFed: "128d embeddings.",
        architectureRole: "Extracts local features for BiLSTM.",
        ablationStudy: "Lacks k=2 bigram filters needed for mock praise and lacks modern Mish activation.",
        inputOutputSpec: { inputType: "FloatTensor (B, 128, 200)", outputType: "FloatTensor (B, 96, 200)" },
        traversalExample: {
            "sampleText": "What a cinematic masterpiece!",
            "steps": [
                {
                    "stage": "Parallel Conv1D Filter Banks (k=3, 5, 7)",
                    "concept": "What are V1 Convolution Banks? Three sets of 1D filters scan the sentence to detect 3-word, 5-word, and 7-word phrases. Uses simple ReLU activation, which turns off all negative values.",
                    "simpleInput": "Embedded sequence transposed to (1 sentence, 128 channels, 200 words).",
                    "simpleProcessing": "Applies 32 filters for each kernel size with ReLU activation: max(0, x).",
                    "simpleOutput": "Three feature maps (32 channels each) capturing basic n-gram phrases.",
                    "impact": "Detects simple phrases, but ReLU completely kills negative gradient signals, missing subtle ironic cues.",
                    "math": {
                        "formula": "F_k = ReLU(Conv1D_k(X)) = max(0, W_k * X + b_k)",
                        "inputTensor": "Embedded word sequence transposed to (batch_size=1, in_channels=128, length=200)",
                        "operation": "Apply 3 independent Conv1D filter banks (32 filters each for k=3, k=5, k=7) with ReLU activation: y = max(0, W * x + b).",
                        "outputTensor": "3 feature map tensors (32 channels each):\n- k=3 (trigrams): (1, 32, 200) Float32\n- k=5 (5-grams): (1, 32, 200) Float32\n- k=7 (7-grams): (1, 32, 200) Float32"
                    },
                    "input": "Embedded word sequence transposed to (batch_size=1, in_channels=128, length=200)",
                    "operation": "Apply 3 independent Conv1D filter banks (32 filters each for k=3, k=5, k=7) with ReLU activation: y = max(0, W * x + b).",
                    "output": "3 feature map tensors (32 channels each):\n- k=3 (trigrams): (1, 32, 200) Float32\n- k=5 (5-grams): (1, 32, 200) Float32\n- k=7 (7-grams): (1, 32, 200) Float32"
                },
                {
                    "stage": "Feature Map Channel Concatenation",
                    "concept": "What is V1 Feature Concatenation? Stacks the 3 filter bank outputs together to create a single 96-channel feature map.",
                    "simpleInput": "Three separate 32-channel feature maps.",
                    "simpleProcessing": "Concatenates along the channel axis (32 + 32 + 32 = 96 channels).",
                    "simpleOutput": "A unified feature map with 96 channels per word.",
                    "impact": "Passes basic phrase features to the attention layer, but lacks the multi-scale depth of V2 and V3 (256 channels).",
                    "math": {
                        "formula": "F_unified = concat([f_k3, f_k5, f_k7], dim=1)",
                        "inputTensor": "Three (1, 32, 200) feature maps",
                        "operation": "Concatenate along channel dimension: cat([f_k3, f_k5, f_k7], dim=1).",
                        "outputTensor": "Unified Feature Tensor (batch_size=1, channels=96, length=200) Float32:\nChannels 0-31 detect 3-word n-grams, channels 32-63 detect 5-word phrases, channels 64-95 detect 7-word clauses (Mean activation = +0.412, variance = 0.814)."
                    },
                    "input": "Three (1, 32, 200) feature maps",
                    "operation": "Concatenate along channel dimension: cat([f_k3, f_k5, f_k7], dim=1).",
                    "output": "Unified Feature Tensor (batch_size=1, channels=96, length=200) Float32:\nChannels 0-31 detect 3-word n-grams, channels 32-63 detect 5-word phrases, channels 64-95 detect 7-word clauses (Mean activation = +0.412, variance = 0.814)."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg"><rect x="30" y="30" width="700" height="60" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/><text x="380" y="65" font-family="'Space Grotesk',sans-serif" font-size="12" font-weight="700" fill="#0f172a" text-anchor="middle">PARALLEL CONV1D (k=3, 5, 7) -> 96 CHANNELS</text></svg>`,
        relatedComponents: ["multiscale_cnn", "bilstm"]
    },

    bahdanau_attention: {
        id: "bahdanau_attention",
        title: "Bahdanau Additive Self-Attention",
        shortName: "Bahdanau Attention",
        category: "Single-Head Sequence Pooling",
        models: ["Model V1 (Baseline)"],
        parameters: "16,512 Parameters (Single Projection Vector)",
        inputShape: "Tensor (batch_size, 200, 128) BiLSTM States",
        outputShape: "Context Vector (batch_size, 128)",
        badge: "V1 Legacy Attention",
        summary: "Classic additive self-attention mechanism computing a single scalar attention weight per time step via tanh projection.",
        beginnerIntuition: "Think of Bahdanau Attention like having a single spotlight in a dark room. It can illuminate one important word, but cannot keep an eye on multiple clues across different corners of the room at the same time.",
        whatIsIt: `Bahdanau Additive Attention is the classical attention mechanism introduced by Bahdanau et al. in 2014. Rather than using multi-head scaled dot products, it computes alignment scores through an additive neural layer:
e_t = v^T * tanh(W_h * h_t + b)
alpha_t = exp(e_t) / sum_k(exp(e_k))
c = sum_t(alpha_t * h_t)
This produces a single 128-dimensional context vector c summarizing the entire review.`,
        howIsItTrained: `Trained jointly with Model V1's single sentiment head via Backpropagation.`,
        dataFed: `Receives Model V1's 128-dimensional BiLSTM hidden outputs.`,
        architectureRole: `Replaced naive max pooling in V1, enabling the baseline model to achieve 88.9% test accuracy on direct IMDB sentiment.`,
        ablationStudy: `Why it was upgraded to Multi-Head Attention in V2 and V3:
- Single Perspective Limit: A single scalar attention weight can only focus on one dominant feature (e.g., attending exclusively to the word "terrible" or "great"). It cannot concurrently attend to modifier-noun contradiction, punctuation, and multi-clause shifts like V3's 4-head attention.`,
        inputOutputSpec: {
            inputType: "Tensor (batch_size, 200, 128)",
            outputType: "Tensor (batch_size, 128)",
            parameters: "16,512 parameters"
        },
        traversalExample: {
            "sampleText": "What an absolute masterpiece of cinematic failure. Brilliant writing!",
            "steps": [
                {
                    "stage": "Additive Score Projection",
                    "concept": "What is Bahdanau Additive Attention? A single-head attention mechanism that computes an energy score for each word to decide how important it is. Because V1 was trained only on literal sentiment, praise words naturally receive the highest scores.",
                    "simpleInput": "BiLSTM sequence states (200 words x 128 dimensions).",
                    "simpleProcessing": "Passes states through a tanh neural layer and projects down to a single energy score for each word.",
                    "simpleOutput": "Energy scores where praise words ('masterpiece', 'Brilliant') score highest.",
                    "impact": "Causes the model to obsess over positive praise words while ignoring the negative punchline.",
                    "math": {
                        "formula": "e_t = v^T * tanh(W_h * h_t + b_h)",
                        "inputTensor": "BiLSTM sequence states H in R^(1 x 200 x 128)",
                        "operation": "Compute additive alignment energy: e_t = v^T * tanh(W_h * h_t + b), where W_h in R^(64 x 128) and v in R^64.",
                        "outputTensor": "Scalar Energy Score Tensor e of shape (1, 200) Float32:\n- t=3 ('masterpiece'): e_3 = +3.42 (High energy)\n- t=8 ('Brilliant'): e_8 = +3.18 (High energy)\n- t=6 ('failure'): e_6 = +1.12 (Low energy)\nPraise words dominate energy projection because V1 was trained purely on literal sentiment."
                    },
                    "input": "BiLSTM sequence states H in R^(1 x 200 x 128)",
                    "operation": "Compute additive alignment energy: e_t = v^T * tanh(W_h * h_t + b), where W_h in R^(64 x 128) and v in R^64.",
                    "output": "Scalar Energy Score Tensor e of shape (1, 200) Float32:\n- t=3 ('masterpiece'): e_3 = +3.42 (High energy)\n- t=8 ('Brilliant'): e_8 = +3.18 (High energy)\n- t=6 ('failure'): e_6 = +1.12 (Low energy)\nPraise words dominate energy projection because V1 was trained purely on literal sentiment."
                },
                {
                    "stage": "Softmax Normalization across Sequence",
                    "concept": "What is Attention Normalization? Converts the raw energy scores into percentages that sum to 100%. The word with the highest percentage gets the lion's share of the model's focus.",
                    "simpleInput": "Energy scores across all 200 words.",
                    "simpleProcessing": "Applies softmax to turn energy scores into probability percentages.",
                    "simpleOutput": "Attention distribution: 'masterpiece' (42.1%), 'Brilliant' (34.8%), 'failure' (11.2%).",
                    "impact": "The single attention spotlight is almost completely monopolized by praise words (77% combined), blinding the model to sarcasm.",
                    "math": {
                        "formula": "alpha_t = exp(e_t) / sum_k(exp(e_k))",
                        "inputTensor": "Energy scores e (1, 200) with padding mask",
                        "operation": "Set padding energy scores to -1e9; normalize with softmax: alpha_t = exp(e_t) / sum_k(exp(e_k)).",
                        "outputTensor": "Single Attention Distribution Vector alpha of shape (1, 200) summing to 1.000:\n- 'masterpiece': 42.1%\n- 'Brilliant': 34.8%\n- 'failure': 11.2%\n- Other tokens: 11.9%\nSingle attention spotlight is heavily monopolized by praise tokens."
                    },
                    "input": "Energy scores e (1, 200) with padding mask",
                    "operation": "Set padding energy scores to -1e9; normalize with softmax: alpha_t = exp(e_t) / sum_k(exp(e_k)).",
                    "output": "Single Attention Distribution Vector alpha of shape (1, 200) summing to 1.000:\n- 'masterpiece': 42.1%\n- 'Brilliant': 34.8%\n- 'failure': 11.2%\n- Other tokens: 11.9%\nSingle attention spotlight is heavily monopolized by praise tokens."
                },
                {
                    "stage": "Weighted Context Vector Synthesis",
                    "concept": "What is Context Synthesis? Multiplies each word vector by its attention percentage and sums them up into a single summary vector.",
                    "simpleInput": "Word vectors and their attention percentages.",
                    "simpleProcessing": "Computes a weighted average of all words in the sentence.",
                    "simpleOutput": "A single 128-number context vector heavily biased toward the praise words.",
                    "impact": "Because the summary reflects praise rather than criticism, V1 naively assumes the review is positive.",
                    "math": {
                        "formula": "c = sum_t (alpha_t * h_t)",
                        "inputTensor": "Attention weights alpha in R^(1 x 200) and hidden states H in R^(1 x 200 x 128)",
                        "operation": "Compute inner product: c = sum_t (alpha_t * h_t).",
                        "outputTensor": "Single Context Vector c in R^(1 x 128) (L2-norm: ||c||_2 = 8.45):\nContext vector is heavily skewed toward positive praise tokens and largely blinds downstream classifier to the negative criticism 'failure'."
                    },
                    "input": "Attention weights alpha in R^(1 x 200) and hidden states H in R^(1 x 200 x 128)",
                    "operation": "Compute inner product: c = sum_t (alpha_t * h_t).",
                    "output": "Single Context Vector c in R^(1 x 128) (L2-norm: ||c||_2 = 8.45):\nContext vector is heavily skewed toward positive praise tokens and largely blinds downstream classifier to the negative criticism 'failure'."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 140" width="100%" height="140" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="35" width="180" height="60" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>
  <text x="110" y="62" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#0f172a" text-anchor="middle">BiLSTM STATES</text>
  <text x="110" y="78" font-family="'Space Mono',monospace" font-size="8.5" fill="#64748b" text-anchor="middle">(B, 200, 128)</text>

  <line x1="200" y1="65" x2="270" y2="65" stroke="#4f46e5" stroke-width="2"/>

  <rect x="270" y="30" width="220" height="70" rx="8" fill="#eef2ff" stroke="#6366f1" stroke-width="1.5"/>
  <text x="380" y="58" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#4338ca" text-anchor="middle">ADDITIVE PROJECTION</text>
  <text x="380" y="74" font-family="'Space Mono',monospace" font-size="8.5" fill="#6366f1" text-anchor="middle">e = v^T * tanh(W*h + b)</text>
  <text x="380" y="90" font-family="'Space Mono',monospace" font-size="8.5" fill="#6366f1" text-anchor="middle">Single Scalar Weight per Token</text>

  <line x1="490" y1="65" x2="560" y2="65" stroke="#4f46e5" stroke-width="2"/>

  <rect x="560" y="35" width="180" height="60" rx="8" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>
  <text x="650" y="62" font-family="'Space Grotesk',sans-serif" font-size="10.5" font-weight="700" fill="#15803d" text-anchor="middle">CONTEXT VECTOR</text>
  <text x="650" y="78" font-family="'Space Mono',monospace" font-size="8.5" fill="#166534" text-anchor="middle">(B, 128) Single Head</text>
</svg>`,
        relatedComponents: ["multihead_attention", "bilstm"]
    },

    single_sentiment_head: {
        id: "single_sentiment_head",
        title: "Single-Task Binary Sentiment Head",
        shortName: "Single Sentiment Head",
        category: "Baseline Classifier",
        models: ["Model V1 (Baseline)"],
        parameters: "16,641 Parameters (Dense 128 + Sigmoid 1)",
        inputShape: "Context Vector (batch_size, 128)",
        outputShape: "Sentiment Probability (batch_size, 1)",
        badge: "V1 Single Objective",
        summary: "Single linear projection followed by sigmoid squashing, predicting direct binary sentiment with no irony or sarcasm awareness.",
        beginnerIntuition: "A single sentiment head only has one lens: is this positive or negative? Because it has no concept of irony or sarcasm, it naively takes words like 'masterpiece' and 'brilliant' literally every single time.",
        whatIsIt: "V1's output projection. Classifies sentiment directly as Positive or Negative.",
        howIsItTrained: "Trained using Binary Cross Entropy (BCE) loss on sentiment labels only.",
        dataFed: "IMDB 50K binary labels.",
        architectureRole: "Generates V1 sentiment classification.",
        ablationStudy: "Cannot detect sarcasm. Systematically fails on mock praise by predicting Positive on sarcastic insults.",
        inputOutputSpec: { inputType: "FloatTensor (B, 128)", outputType: "Float in [0, 1]" },
        traversalExample: {
            "sampleText": "Single-task classification on sarcastic praise",
            "steps": [
                {
                    "stage": "Linear Projection to Unbounded Logit",
                    "concept": "What is Single-Task Projection? Model V1 has only one job: classify sentiment as positive or negative. It has no auxiliary sarcasm head to warn it about irony. The context vector is directly projected to a single sentiment logit.",
                    "simpleInput": "The context vector c (128 numbers biased toward praise words).",
                    "simpleProcessing": "Applies 30% dropout and multiplies by linear weights to compute a raw positive/negative logit.",
                    "simpleOutput": "Positive Logit = +1.15 (leans toward positive decision).",
                    "impact": "Because the model lacked sarcasm awareness, deceptive praise directly pushes the logit into positive territory.",
                    "math": {
                        "formula": "logit = W_out * Dropout_0.3(c) + b_out",
                        "inputTensor": "Context vector c in R^(1 x 128)",
                        "operation": "Apply dropout (p=0.3) followed by linear projection: logit = W_out * c + b_out, where W_out in R^(1 x 128).",
                        "outputTensor": "Scalar Sentiment Logit = +1.15:\nBecause context vector c was heavily biased toward praise words ('masterpiece', 'Brilliant'), the projection maps to positive decision space (+1.15 standard deviations above threshold 0)."
                    },
                    "input": "Context vector c in R^(1 x 128)",
                    "operation": "Apply dropout (p=0.3) followed by linear projection: logit = W_out * c + b_out, where W_out in R^(1 x 128).",
                    "output": "Scalar Sentiment Logit = +1.15:\nBecause context vector c was heavily biased toward praise words ('masterpiece', 'Brilliant'), the projection maps to positive decision space (+1.15 standard deviations above threshold 0)."
                },
                {
                    "stage": "Sigmoidal Binary Classification",
                    "concept": "What is Binary Classification Failure? Passes the positive logit through a sigmoid function to produce the final classification. Since V1 cannot detect irony, it confidently declares sarcastic reviews as positive.",
                    "simpleInput": "Positive logit = +1.15.",
                    "simpleProcessing": "Calculates sigmoid(1.15) = 75.9%.",
                    "simpleOutput": "Final Prediction: POSITIVE (76% Positive, 24% Negative).",
                    "impact": "Exposes the classic failure of baseline single-task models: mock praise ('Brilliant writing!') completely fools the classifier.",
                    "math": {
                        "formula": "p_pos = 1 / (1 + exp(-1.15))",
                        "inputTensor": "Scalar logit = +1.15",
                        "operation": "Apply standard sigmoid: p_pos = 1 / (1 + e^(-1.15)).",
                        "outputTensor": "Calibrated Positive Probability = 0.759 (76% Positive, 24% Negative):\nFinal Classification: POSITIVE.\nClassic Single-Task Failure Mode: V1 naively misclassifies scathing mock praise as genuine praise because it possesses no sarcasm head or gated inversion."
                    },
                    "input": "Scalar logit = +1.15",
                    "operation": "Apply standard sigmoid: p_pos = 1 / (1 + e^(-1.15)).",
                    "output": "Calibrated Positive Probability = 0.759 (76% Positive, 24% Negative):\nFinal Classification: POSITIVE.\nClassic Single-Task Failure Mode: V1 naively misclassifies scathing mock praise as genuine praise because it possesses no sarcasm head or gated inversion."
                }
            ]
        },
        diagram: `<svg viewBox="0 0 760 120" width="100%" height="120" xmlns="http://www.w3.org/2000/svg"><rect x="30" y="30" width="700" height="60" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/><text x="380" y="65" font-family="'Space Grotesk',sans-serif" font-size="12" font-weight="700" fill="#0f172a" text-anchor="middle">LINEAR PROJECTION (128 -> 1) -> SIGMOID</text></svg>`,
        relatedComponents: ["multitask_heads", "bahdanau_attention"]
    }
};
