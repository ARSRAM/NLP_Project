import re
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.base import BaseEstimator, TransformerMixin

# =============================================================================
# 1.  FEATURE BLOCK 1 + 2 — TF-IDF (word + char)
# =============================================================================

def build_tfidf_word(max_features=15000):
    """
    Word unigrams with sublinear TF scaling.
    Captures topic-specific vocabulary — the strongest single signal.
    """
    return TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        ngram_range=(1, 2),       # unigrams + bigrams for phrase capture
        min_df=2,
        strip_accents="unicode",
    )


def build_tfidf_char(max_features=10000):
    """
    Character 3-5 gram TF-IDF.

    INNOVATION — FREE POS PROXY:
    Character n-grams implicitly encode morphological patterns that
    strongly correlate with POS tags without any tagging:

      "-ing" suffix  → present participle (VBG)
      "-ed"  suffix  → past tense verb (VBD) or adjective
      "-tion" suffix → nominalisation (NN)
      "-ly"  suffix  → adverb (RB)
      "Xxx"  pattern → proper noun (NNP)

    Reuters-specific: financial suffixes like "-tion" (acquisition,
    inflation), "-ity" (liquidity, volatility) are strong topic signals
    that word-level TF-IDF misses across surface variations.
    """
    return TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        analyzer="char_wb",        # char n-grams within word boundaries
        ngram_range=(3, 5),
        min_df=3,
    )


# =============================================================================
# 2.  FEATURE BLOCK 3 — Synthetic Linguistic Proxy (15 features)
# =============================================================================

class SyntheticLinguisticProxy(BaseEstimator, TransformerMixin):
    """
    CORE INNOVATION: 15-dimensional hand-crafted proxy feature vector
    that approximates POS and NER signals using only regex + statistics.

    Each feature is designed to replicate a specific linguistic dimension:

    SYNTAX PROXIES (approximate POS histogram):
      f01  Avg word length         — proxy for noun density (nouns are longer)
      f02  Avg sentence length     — proxy for syntactic complexity
      f03  Sentence length std     — variance in sentence structure
      f04  Comma rate              — proxy for clause count / coordination (CC)
      f05  Semicolon + colon rate  — formal / technical writing style
      f06  Quote rate              — reported speech, direct quotation
      f07  Capital word ratio      — proxy for proper noun (NNP) density
      f08  ALL-CAPS word ratio     — abbreviations, acronyms (e.g. NATO, GDP)

    ENTITY PROXIES (approximate NER histogram):
      f09  Number token ratio      — CARDINAL / MONEY / DATE entity proxy
      f10  Currency symbol density — MONEY entity proxy ($, £, ¥, %)
      f11  Percentage density      — statistical / financial reporting signal
      f12  Consecutive caps ratio  — multi-word proper noun proxy (e.g. "New York")

    LEXICAL STYLE:
      f13  Type-token ratio        — vocabulary richness (shorter = more repetitive)
      f14  Short word ratio        — function word density (DT, IN, CC proxies)
      f15  Long word ratio         — technical / domain-specific term density

    WHY THIS WORKS:
    Reuters topics have highly distinctive linguistic signatures:
      - 'earn'   → high numbers, currency symbols, short sentences
      - 'acq'    → high proper nouns, organisation names, formal syntax
      - 'grain'  → medium complexity, geographic names, quantity terms
      - 'crude'  → high numbers, percentage terms, technical vocabulary
    These proxies capture that variance without any tagger.
    """

    def fit(self, X, y=None):
        return self

    def _extract(self, text: str) -> np.ndarray:
        # Tokenise cheaply
        words    = text.split()
        n_words  = max(len(words), 1)
        chars    = text.replace(" ", "")
        n_chars  = max(len(chars), 1)

        # Sentences (split on . ! ?)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        n_sents   = max(len(sentences), 1)
        sent_lens = [len(s.split()) for s in sentences]

        # f01 avg word length
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        # f02 avg sentence length (words)
        avg_sent_len = np.mean(sent_lens) if sent_lens else 0

        # f03 sentence length std
        std_sent_len = np.std(sent_lens) if len(sent_lens) > 1 else 0

        # f04 comma rate
        comma_rate = text.count(",") / n_words

        # f05 semicolon + colon rate
        semi_rate = (text.count(";") + text.count(":")) / n_words

        # f06 quote rate
        quote_rate = (text.count('"') + text.count("'")) / n_words

        # f07 capital word ratio (words starting with uppercase, not sentence-start)
        cap_words = sum(1 for w in words if w and w[0].isupper() and w.isalpha())
        cap_ratio = cap_words / n_words

        # f08 ALL-CAPS ratio
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        caps_ratio = caps_words / n_words

        # f09 number token ratio
        num_words  = sum(1 for w in words if re.search(r"\d", w))
        num_ratio  = num_words / n_words

        # f10 currency symbol density
        currency   = len(re.findall(r"[\$£€¥]", text))
        curr_ratio = currency / n_words

        # f11 percentage density
        pct_count  = len(re.findall(r"\d+\.?\d*\s*%|pct", text, re.IGNORECASE))
        pct_ratio  = pct_count / n_words

        # f12 consecutive capitalised words (multi-word proper noun proxy)
        consec_caps = len(re.findall(r"(?:[A-Z][a-z]+\s){2,}[A-Z][a-z]+", text))
        consec_ratio = consec_caps / n_sents

        # f13 type-token ratio (capped at 200 words for fairness)
        sample   = words[:200]
        ttr      = len(set(w.lower() for w in sample)) / max(len(sample), 1)

        # f14 short word ratio (≤3 chars — function word proxy)
        short_ratio = sum(1 for w in words if len(w) <= 3) / n_words

        # f15 long word ratio (≥8 chars — technical term proxy)
        long_ratio  = sum(1 for w in words if len(w) >= 8) / n_words

        return np.array([
            avg_word_len, avg_sent_len, std_sent_len,
            comma_rate, semi_rate, quote_rate,
            cap_ratio, caps_ratio, num_ratio,
            curr_ratio, pct_ratio, consec_ratio,
            ttr, short_ratio, long_ratio
        ], dtype=np.float32)

    def transform(self, X):
        return np.vstack([self._extract(doc) for doc in X])


# =============================================================================
# 3.  FEATURE BLOCK 4 — Topic Affinity (Label Co-occurrence)
# =============================================================================

class TopicAffinityFeature(BaseEstimator, TransformerMixin):
    """
    INNOVATION: Topic affinity via label co-occurrence centroids.

    Learns a TF-IDF centroid for each topic from training documents,
    then represents each document as its cosine similarity to all
    topic centroids. This gives the model a soft "which topics does
    this document sound like?" signal before the classifier sees it.

    For Reuters multi-label setting this is especially powerful:
    topics co-occur in structured patterns (wheat+grain, usa+trade)
    and the affinity vector captures that structure.
    """

    def __init__(self, max_features=5000):
        self.max_features = max_features
        self._tfidf     = TfidfVectorizer(max_features=max_features,
                                          sublinear_tf=True, min_df=2)
        self._centroids = None

    def fit(self, X, y_bin):
        mat = self._tfidf.fit_transform(X)           # (n_docs, vocab)
        n_labels = y_bin.shape[1]
        V = mat.shape[1]
        centroids = np.zeros((n_labels, V), dtype=np.float32)
        for j in range(n_labels):
            mask = y_bin[:, j].astype(bool)
            if mask.sum() > 0:
                c = np.asarray(mat[mask].mean(axis=0)).ravel()
                norm = np.linalg.norm(c)
                centroids[j] = c / norm if norm > 0 else c
        self._centroids = centroids
        return self

    def transform(self, X):
        mat = self._tfidf.transform(X)               # (n_docs, vocab)
        sim = mat.dot(self._centroids.T)              # (n_docs, n_labels)
        return np.asarray(sim)


# =============================================================================
# 4.  FEATURE BLOCK 5 — Meta-Group Affinity (5-dim)
# =============================================================================

META_GROUPS = {
    "finance":   ["earn","acq","money-fx","interest","trade","dlr","gnp",
                  "money-supply","bop","reserves","debt","ipi","wpi","cpi"],
    "commodity": ["grain","wheat","corn","coffee","sugar","crude","cotton",
                  "rubber","rice","oilseed","soybean","livestock","gold",
                  "silver","copper","nat-gas","fuel","cocoa","oilseed"],
    "geography": ["usa","uk","canada","japan","west-germany","france","brazil",
                  "australia","china","india","ussr","spain","mexico","italy"],
    "corporate": ["acq","earn","jobs","housing","retail","income","insolvency"],
    "politics":  ["election","military","law","ec","opec","war"],
}

class MetaGroupAffinityFeature(BaseEstimator, TransformerMixin):
    """
    5-dimensional coarse domain signal — cosine similarity of each
    document to each meta-group centroid (finance, commodity,
    geography, corporate, politics).

    This gives the classifier a high-level prior before it sees any
    per-label evidence, improving performance on rare topics that
    belong to a well-represented group.
    """

    def __init__(self, max_features=3000):
        self.max_features = max_features
        self._tfidf = TfidfVectorizer(max_features=max_features,
                                      sublinear_tf=True, min_df=2)
        self._centroids = []

    def fit(self, X, y_bin, label_names):
        mat = self._tfidf.fit_transform(X)
        name2col = {n: i for i, n in enumerate(label_names)}
        self._centroids = []
        for group_topics in META_GROUPS.values():
            cols = [name2col[t] for t in group_topics if t in name2col]
            if not cols:
                self._centroids.append(np.zeros(mat.shape[1], dtype=np.float32))
                continue
            mask = np.zeros(y_bin.shape[0], dtype=bool)
            for c in cols:
                mask |= y_bin[:, c].astype(bool)
            if not mask.any():
                self._centroids.append(np.zeros(mat.shape[1], dtype=np.float32))
                continue
            c = np.asarray(mat[mask].mean(axis=0)).ravel().astype(np.float32)
            norm = np.linalg.norm(c)
            self._centroids.append(c / norm if norm > 0 else c)
        return self

    def transform(self, X):
        mat = self._tfidf.transform(X)
        sims = [np.asarray(mat.dot(c)).ravel() for c in self._centroids]
        return np.column_stack(sims)


# =============================================================================
# 5.  PIPELINE BUILDER
# =============================================================================

def build_features(X_train, X_test, y_train_bin, mlb, config, verbose=True):
    """
    Build feature matrix for a given ablation config.
    All blocks stay sparse; dense blocks are wrapped in csr_matrix.
    """
    blocks_tr, blocks_te = [], []

    def log(msg):
        if verbose: print(f"  {msg}")

    # Block 1 — TF-IDF word
    log("[1] TF-IDF word n-grams...")
    tw = build_tfidf_word(max_features=config.get("tfidf_word", 15000))
    tr = normalize(tw.fit_transform(X_train), norm="l2")
    te = normalize(tw.transform(X_test),      norm="l2")
    blocks_tr.append(tr); blocks_te.append(te)
    log(f"    → {tr.shape[1]} features")

    # Block 2 — TF-IDF char (syntax proxy)
    if config.get("char_tfidf", True):
        log("[2] TF-IDF char n-grams (syntax proxy)...")
        tc = build_tfidf_char(max_features=config.get("tfidf_char", 10000))
        tr = normalize(tc.fit_transform(X_train), norm="l2")
        te = normalize(tc.transform(X_test),      norm="l2")
        blocks_tr.append(tr); blocks_te.append(te)
        log(f"    → {tr.shape[1]} features")

    # Block 3 — Synthetic linguistic proxy
    if config.get("synth_proxy", True):
        log("[3] Synthetic linguistic proxy (15-dim)...")
        sp = SyntheticLinguisticProxy()
        tr_d = sp.fit(X_train).transform(X_train)
        te_d = sp.transform(X_test)
        sc = StandardScaler()
        tr_d = sc.fit_transform(tr_d)
        te_d = sc.transform(te_d)
        blocks_tr.append(csr_matrix(tr_d))
        blocks_te.append(csr_matrix(te_d))
        log(f"    → {tr_d.shape[1]} features")

    # Block 4 — Topic affinity
    if config.get("topic_affinity", True):
        log("[4] Topic affinity features...")
        ta = TopicAffinityFeature(max_features=5000)
        tr_d = ta.fit(X_train, y_train_bin).transform(X_train)
        te_d = ta.transform(X_test)
        sc = StandardScaler()
        tr_d = sc.fit_transform(tr_d)
        te_d = sc.transform(te_d)
        blocks_tr.append(csr_matrix(tr_d))
        blocks_te.append(csr_matrix(te_d))
        log(f"    → {tr_d.shape[1]} features")

    # Block 5 — Meta group affinity
    if config.get("meta_affinity", True):
        log("[5] Meta-group affinity (5-dim)...")
        mg = MetaGroupAffinityFeature(max_features=3000)
        mg.fit(X_train, y_train_bin, list(mlb.classes_))
        tr_d = mg.transform(X_train)
        te_d = mg.transform(X_test)
        sc = StandardScaler()
        tr_d = sc.fit_transform(tr_d)
        te_d = sc.transform(te_d)
        blocks_tr.append(csr_matrix(tr_d))
        blocks_te.append(csr_matrix(te_d))
        log(f"    → {tr_d.shape[1]} features")

    X_tr = hstack(blocks_tr, format="csr")
    X_te = hstack(blocks_te, format="csr")
    log(f"  Total: {X_tr.shape[1]} features\n")
    return X_tr, X_te
