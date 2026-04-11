import time
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score

from features import build_features

# =============================================================================
# 6.  CLASSIFIER + PREDICTION
# =============================================================================

def train_clf(X_train, y_train_bin, C=1.0):
    clf = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=1000, solver="saga", random_state=42),
        n_jobs=-1,
    )
    clf.fit(X_train, y_train_bin)
    return clf


def predict(clf, X, threshold=0.3):
    """Threshold-tuned prediction — avoids the high-P/low-R collapse."""
    proba = clf.predict_proba(X)
    pred  = (proba >= threshold).astype(int)
    # Always assign at least one label
    no_label = pred.sum(axis=1) == 0
    if no_label.any():
        pred[no_label, proba[no_label].argmax(axis=1)] = 1
    return pred


# =============================================================================
# 7.  EVALUATION
# =============================================================================

def evaluate(clf, X_te, y_te, mlb, label="Test"):
    y_pred = predict(clf, X_te)
    metrics = {
        "Micro F1":    f1_score(y_te, y_pred, average="micro",     zero_division=0),
        "Macro F1":    f1_score(y_te, y_pred, average="macro",     zero_division=0),
        "Weighted F1": f1_score(y_te, y_pred, average="weighted",  zero_division=0),
        "Micro P":     precision_score(y_te, y_pred, average="micro", zero_division=0),
        "Micro R":     recall_score(y_te, y_pred,    average="micro", zero_division=0),
        "Hamming":     hamming_loss(y_te, y_pred),
    }
    print(f"\n{'='*50}\n  {label} Results\n{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:<14}: {v:.4f}")

    per_label_f1 = f1_score(y_te, y_pred, average=None, zero_division=0)
    per_label_p  = precision_score(y_te, y_pred, average=None, zero_division=0)
    per_label_r  = recall_score(y_te, y_pred, average=None, zero_division=0)
    support      = y_te.sum(axis=0)
    label_df = pd.DataFrame({
        "label":     mlb.classes_,
        "precision": per_label_p,
        "recall":    per_label_r,
        "f1":        per_label_f1,
        "support":   support,
    }).sort_values("f1", ascending=False)
    return metrics, label_df


# =============================================================================
# 8.  ABLATION
# =============================================================================

ABLATION_CONFIGS = [
    ("TF-IDF word only (baseline)",
     {"char_tfidf": False, "synth_proxy": False,
      "topic_affinity": False, "meta_affinity": False}),

    ("+ Char TF-IDF (syntax proxy)",
     {"char_tfidf": True,  "synth_proxy": False,
      "topic_affinity": False, "meta_affinity": False}),

    ("+ Synthetic proxy (15-dim)",
     {"char_tfidf": True,  "synth_proxy": True,
      "topic_affinity": False, "meta_affinity": False}),

    ("+ Topic affinity",
     {"char_tfidf": True,  "synth_proxy": True,
      "topic_affinity": True,  "meta_affinity": False}),

    ("Full model (+ meta affinity)",
     {"char_tfidf": True,  "synth_proxy": True,
      "topic_affinity": True,  "meta_affinity": True}),
]


def run_ablation(train, test, y_tr_bin, y_te_bin, mlb):
    X_tr = list(train["text_clean"])
    X_te = list(test["text_clean"])
    rows = []
    for name, cfg in ABLATION_CONFIGS:
        print(f"\n{'─'*50}")
        print(f"  Config: {name}")
        print(f"{'─'*50}")
        t0 = time.time()
        Xtr, Xte = build_features(X_tr, X_te, y_tr_bin, mlb, cfg)
        clf = train_clf(Xtr, y_tr_bin)
        metrics, _ = evaluate(clf, Xte, y_te_bin, mlb, label=name)
        elapsed = time.time() - t0
        rows.append({"Config": name, **{k: round(v, 4) for k, v in metrics.items()},
                     "Time(s)": round(elapsed, 1)})
    return pd.DataFrame(rows)


# =============================================================================
# 9.  FREQUENCY-STRATIFIED ANALYSIS
# =============================================================================

def stratified_analysis(label_df, train):
    freq = Counter(t for labels in train["label_list"] for t in labels)
    freq_df = pd.DataFrame({"label": list(freq.keys()),
                             "train_count": list(freq.values())})
    merged = label_df.merge(freq_df, on="label", how="left").fillna(0)

    def bucket(n):
        if n >= 500: return "Very Common (≥500)"
        if n >= 100: return "Common (100–499)"
        return "Rare (<100)"

    merged["group"] = merged["train_count"].apply(bucket)
    print("\n  Frequency-Stratified F1:")
    print(f"  {'Group':<22}  {'Avg F1':>7}  {'N':>5}")
    print("  " + "─" * 38)
    for g in ["Very Common (≥500)", "Common (100–499)", "Rare (<100)"]:
        sub = merged[merged["group"] == g]
        if len(sub):
            print(f"  {g:<22}  {sub['f1'].mean():>7.4f}  {len(sub):>5}")
    return merged
