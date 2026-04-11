"""
=============================================================================
LinguaSynth-Light: Proxy-Based Linguistic Features for Reuters Classification
Project: Syntactic Features for Downstream Tasks on Reuters Dataset
=============================================================================

CORE INNOVATION:
  Instead of expensive POS taggers and NER pipelines, we introduce
  "Synthetic Linguistic Proxies" — five cheap regex/character-level
  features that approximate what POS and NER capture, derived in
  milliseconds with zero external NLP tools.

  Proxy → What it approximates
  ─────────────────────────────────────────────────────────
  Capitalisation ratio    → NER (proper nouns, organisations)
  Number density          → NER (CARDINAL, MONEY, DATE entities)
  Punctuation profile     → POS (sentence complexity, syntax style)
  Word-shape distribution → POS (noun-adj-verb density proxies)
  Readability metrics     → Syntactic complexity (clause depth)

FEATURE BLOCKS (all sklearn, no extra installs):
  1. TF-IDF word unigrams       (lexical)
  2. TF-IDF char 3-5 grams      (subword morphology — free POS proxy)
  3. Synthetic syntax proxy     (15-dim hand-crafted vector)
  4. Topic affinity             (label co-occurrence graph — innovation)
  5. Meta-group affinity        (5-dim domain cluster signal)

Requirements: sklearn, pandas, numpy, scipy  (all standard)


Usage:
    python Linguasynth_reuters.py
=============================================================================
"""

import ast
import warnings
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from features import build_features
from evaluation import train_clf, evaluate, run_ablation, stratified_analysis

warnings.filterwarnings("ignore")


# =============================================================================
# 0.  DATA LOADING
# =============================================================================

def load_data(path="data_clean.csv"):
    df = pd.read_csv(path)
    df["label_list"] = df["topic_list"].apply(ast.literal_eval)
    train = df[df["split"] == "TRAIN"].reset_index(drop=True)
    test  = df[df["split"] == "TEST"].reset_index(drop=True)
    print(f"Train: {len(train)}  |  Test: {len(test)}")
    print(f"Labels: {df['label_list'].explode().nunique()} unique topics")
    print(f"Avg labels/doc: {df['num_labels'].mean():.2f}\n")
    return train, test


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    print("=" * 50)
    print("  LinguaSynth-Light  —  Reuters Multi-Label")
    print("  Proxy-Based Linguistic Feature Fusion")
    print("=" * 50)

    train, test = load_data("data_clean.csv")

    mlb = MultiLabelBinarizer(sparse_output=False)
    y_tr = mlb.fit_transform(train["label_list"])
    y_te = mlb.transform(test["label_list"])
    print(f"Label matrix: {y_tr.shape}  →  {y_te.shape}\n")

    # --- Ablation ---
    print("=" * 50 + "\n  ABLATION STUDY\n" + "=" * 50)
    abl = run_ablation(train, test, y_tr, y_te, mlb)
    print("\n\n" + "=" * 50 + "\n  ABLATION SUMMARY\n" + "=" * 50)
    print(abl.to_string(index=False))
    abl.to_csv("ablation_summary.csv", index=False)

    # --- Full model per-label results ---
    print("\n" + "=" * 50 + "\n  FULL MODEL — PER-LABEL RESULTS\n" + "=" * 50)
    X_tr = list(train["text_clean"])
    X_te = list(test["text_clean"])
    Xtr, Xte = build_features(X_tr, X_te, y_tr, mlb,
                               {"char_tfidf": True, "synth_proxy": True,
                                "topic_affinity": True, "meta_affinity": True})
    clf = train_clf(Xtr, y_tr)
    _, label_df = evaluate(clf, Xte, y_te, mlb)

    print("\n  Top 15 labels by F1:")
    print(label_df.head(15).to_string(index=False))
    label_df.to_csv("results_per_label.csv", index=False)

    # --- Stratified analysis ---
    print("\n" + "=" * 50 + "\n  FREQUENCY-STRATIFIED ANALYSIS\n" + "=" * 50)
    strat = stratified_analysis(label_df, train)
    strat.to_csv("results_stratified.csv", index=False)

    print("\n\nOutput files: ablation_summary.csv, results_per_label.csv, results_stratified.csv")
    print("Done.")


if __name__ == "__main__":
    main()