"""
eval/compare.py
───────────────
Focused evaluation script for T-JEPA research.

Compares:
    1. T-JEPA (your trained model)      ← your checkpoint
    2. DistilBERT vanilla (MLM baseline)← pretrained, no extra training
    3. T-JEPA ablations                 ← different configs you trained

Tasks:
    A. Semantic Similarity  — cosine sim between paraphrase pairs vs unrelated pairs
    B. Linear Probe         — logistic regression on frozen embeddings (AG News, 4-class)
    C. K-Means Clustering   — ARI + NMI + Silhouette on AG News topics

Output:
    Printed table only — clean, copy-pasteable into your paper/report.

Usage:
    # Basic: T-JEPA vs DistilBERT baseline only
    python eval/compare.py --ckpt checkpoints/step_050000.pt

    # With ablations (point to a folder of checkpoints):
    python eval/compare.py --ckpt checkpoints/step_050000.pt \\
                           --ablation-dir checkpoints/ablations/

    # Quick smoke test (small data, fast):
    python eval/compare.py --ckpt checkpoints/step_050000.pt --quick

Install deps:
    pip install datasets scikit-learn scipy transformers torch
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EMBEDDING EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_embeddings(
    texts: list,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
    pool: str = "cls",
) -> np.ndarray:
    """
    Extract sentence-level embeddings from any transformer model.

    Pooling options:
        "cls"  — use the [CLS] token representation (position 0)
                 Standard for BERT-family models. Aggregates full sentence.
        "mean" — average all non-padding token representations
                 More robust when [CLS] wasn't explicitly trained.

    Args:
        texts      : list of raw text strings
        model      : any HuggingFace AutoModel (already on device, eval mode)
        tokenizer  : matching tokenizer
        device     : torch device
        batch_size : how many sentences to encode at once
        max_length : truncate/pad to this token length
        pool       : "cls" or "mean"

    Returns:
        np.ndarray of shape (N, hidden_size)
    """
    model.eval()
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids      = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            hidden = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state                         # (B, T, H)

            if pool == "cls":
                emb = hidden[:, 0, :]                  # (B, H)
            else:
                # Mean of real tokens (ignore padding)
                mask = attention_mask.unsqueeze(-1).float()
                emb  = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)            # (N, H)


def load_tjepa_teacher(ckpt_path: str, cfg: dict, device: torch.device):
    """
    Load T-JEPA's teacher encoder from a checkpoint.

    We always use the TEACHER (not student) for evaluation because:
    - Teacher saw clean, unmasked text throughout training
    - Teacher is an EMA average of all past student states → more stable
    - Teacher is what you'd deploy for downstream tasks in production

    Returns:
        (model, tokenizer) both ready for inference
    """
    # Import your encoder — adjust path if needed
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models.encoder import TextEncoder

    print(f"  Loading T-JEPA teacher ← {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    encoder = TextEncoder(cfg["model"]["encoder_name"], cfg["model"]["proj_dim"])

    # Load teacher weights — key must match what helper.py saved
    encoder.load_state_dict(ckpt["teacher_state_dict"])
    encoder.eval().to(device)

    # We access the raw transformer (before projection) for fair comparison
    # with DistilBERT — both produce 768-dim vectors this way
    # To use projected latents instead, remove ".encoder" below
    raw_encoder = encoder.encoder

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    return raw_encoder, tokenizer


def load_distilbert_baseline(device: torch.device):
    """
    Load vanilla DistilBERT — the MLM baseline.

    This is DistilBERT exactly as HuggingFace releases it, with no
    additional training. Any improvement T-JEPA shows over this is
    attributable to your SSL pretraining objective.

    Returns:
        (model, tokenizer)
    """
    model_name = "distilbert-base-uncased"
    print(f"  Loading DistilBERT baseline ← {model_name} (HuggingFace)")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).eval().to(device)
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EVALUATION DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_ag_news(n_per_class: int = 500) -> tuple:
    """
    Load AG News dataset for linear probe + clustering tasks.

    AG News has 4 topic classes:
        0 = World News
        1 = Sports
        2 = Business
        3 = Science / Technology

    We use the test split (7,600 samples) and subsample evenly per class
    so the evaluation is balanced.

    Args:
        n_per_class : samples per class to use (total = 4 × n_per_class)

    Returns:
        (texts, labels, label_names) — list of str, np.ndarray of ints, list of str
    """
    from datasets import load_dataset

    print("  Loading AG News...")
    ds = load_dataset("ag_news", split="test")
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    texts, labels = [], []
    for cls in range(4):
        cls_samples = [(x["text"], x["label"]) for x in ds if x["label"] == cls]
        cls_samples = cls_samples[:n_per_class]
        for text, label in cls_samples:
            texts.append(text)
            labels.append(label)

    # Shuffle so classes are interleaved (important for k-means init)
    idx = np.random.permutation(len(texts))
    texts  = [texts[i] for i in idx]
    labels = np.array([labels[i] for i in idx])

    print(f"    {len(texts)} samples loaded ({n_per_class} per class)")
    return texts, labels, label_names


def load_mrpc_pairs(n: int = 300) -> tuple:
    """
    Load MRPC paraphrase pairs for semantic similarity task.

    MRPC (Microsoft Research Paraphrase Corpus) contains sentence pairs
    labelled as paraphrase (1) or not-paraphrase (0).

    We use it to measure:
        pos_sim : avg cosine similarity between paraphrase pairs
        neg_sim : avg cosine similarity between non-paraphrase pairs
        gap     : pos_sim − neg_sim  (KEY METRIC — higher = better model)

    A good model should place paraphrases close together in latent space
    and unrelated sentences far apart.

    Args:
        n : number of positive and negative pairs each

    Returns:
        (pos_pairs_a, pos_pairs_b, neg_pairs_a, neg_pairs_b)
        Each is a list of n text strings
    """
    from datasets import load_dataset

    print("  Loading MRPC pairs...")
    ds = load_dataset("glue", "mrpc", split="test")

    pos_a, pos_b, neg_a, neg_b = [], [], [], []
    for x in ds:
        if x["label"] == 1 and len(pos_a) < n:
            pos_a.append(x["sentence1"])
            pos_b.append(x["sentence2"])
        elif x["label"] == 0 and len(neg_a) < n:
            neg_a.append(x["sentence1"])
            neg_b.append(x["sentence2"])

    actual_n = min(len(pos_a), len(neg_a))
    print(f"    {actual_n} positive pairs, {actual_n} negative pairs loaded")
    return pos_a[:actual_n], pos_b[:actual_n], neg_a[:actual_n], neg_b[:actual_n]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EVALUATION TASKS
# ══════════════════════════════════════════════════════════════════════════════

def task_semantic_similarity(
    embs_pos_a: np.ndarray,
    embs_pos_b: np.ndarray,
    embs_neg_a: np.ndarray,
    embs_neg_b: np.ndarray,
) -> dict:
    """
    Task A: Semantic Similarity via Cosine Distance.

    Measures whether the model assigns high similarity to paraphrase pairs
    and low similarity to unrelated pairs.

    The GAP (pos_sim − neg_sim) is the most important single number here.
    A random embedding space has gap ≈ 0. A good model should have gap > 0.3.

    Args:
        embs_pos_a/b : (N, D) embeddings of positive (paraphrase) pairs
        embs_neg_a/b : (N, D) embeddings of negative (unrelated) pairs

    Returns:
        dict with pos_sim, neg_sim, gap, pos_std, neg_std
    """
    def cosine_sim(a, b):
        a = normalize(a, norm="l2")
        b = normalize(b, norm="l2")
        return (a * b).sum(axis=1)          # (N,) elementwise dot product

    pos_sims = cosine_sim(embs_pos_a, embs_pos_b)
    neg_sims = cosine_sim(embs_neg_a, embs_neg_b)

    return {
        "pos_sim": float(pos_sims.mean()),
        "neg_sim": float(neg_sims.mean()),
        "gap":     float(pos_sims.mean() - neg_sims.mean()),   # KEY METRIC
        "pos_std": float(pos_sims.std()),
        "neg_std": float(neg_sims.std()),
    }


def task_linear_probe(
    train_embs:   np.ndarray,
    train_labels: np.ndarray,
    test_embs:    np.ndarray,
    test_labels:  np.ndarray,
) -> dict:
    """
    Task B: Linear Probe Classification.

    Trains a logistic regression on frozen embeddings. No neural network,
    no fine-tuning — just a single linear layer on top of fixed features.

    This is the standard SSL evaluation protocol (used in SimCLR, BYOL,
    I-JEPA papers). A higher accuracy means the model has encoded more
    task-relevant information in a linearly separable way.

    Rule of thumb for AG News (4 classes):
        < 0.70 : poor representations
        0.70–0.80 : acceptable
        > 0.80 : good SSL pretraining

    Args:
        train_embs/labels : training split embeddings + integer class labels
        test_embs/labels  : test split embeddings + labels

    Returns:
        dict with accuracy, num_classes, num_train, num_test
    """
    train_norm = normalize(train_embs, norm="l2")
    test_norm  = normalize(test_embs,  norm="l2")

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        multi_class="auto",
        n_jobs=-1,
    )
    clf.fit(train_norm, train_labels)
    preds = clf.predict(test_norm)

    return {
        "accuracy":    float(accuracy_score(test_labels, preds)),
        "num_classes": int(len(np.unique(train_labels))),
        "num_train":   int(len(train_labels)),
        "num_test":    int(len(test_labels)),
    }


def task_clustering(
    embs:         np.ndarray,
    true_labels:  np.ndarray,
    n_clusters:   int = None,
) -> dict:
    """
    Task C: K-Means Clustering Quality.

    Runs K-Means on embeddings and compares the resulting clusters to the
    true class labels using three metrics:

        ARI (Adjusted Rand Index):
            Measures overlap between predicted clusters and true labels.
            0 = random, 1 = perfect match. Adjusted for chance.

        NMI (Normalized Mutual Information):
            Measures shared information between clusters and true labels.
            0 = no information shared, 1 = perfect.

        Silhouette Score:
            Measures how compact and separated the clusters are.
            -1 to +1. Higher = tighter clusters with clear boundaries.

    Rule of thumb:
        ARI > 0.5 : model has learned topic structure without supervision
        ARI > 0.7 : excellent — representations closely match human categories

    Args:
        embs        : (N, D) embeddings
        true_labels : (N,) integer class labels
        n_clusters  : defaults to number of unique classes

    Returns:
        dict with ari, nmi, silhouette
    """
    if n_clusters is None:
        n_clusters = len(np.unique(true_labels))

    embs_norm = normalize(embs, norm="l2")

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    pred_labels = km.fit_predict(embs_norm)

    # Silhouette is expensive on large N — subsample to 3000 max
    n_sil = min(len(embs), 3000)
    sil_idx = np.random.choice(len(embs), n_sil, replace=False)

    return {
        "ari":        float(adjusted_rand_score(true_labels, pred_labels)),
        "nmi":        float(normalized_mutual_info_score(true_labels, pred_labels)),
        "silhouette": float(silhouette_score(embs_norm[sil_idx], pred_labels[sil_idx])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ABLATION CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

# Define what each ablation experiment changes.
# Key = experiment name shown in the table
# "ckpt_subdir" = folder inside --ablation-dir containing that experiment's checkpoint
# "description" = shown in the table for clarity

ABLATION_REGISTRY = {
    "JEPA (mask=10%)": {
        "ckpt_subdir": "mask_10pct",
        "description": "Mask ratio 0.10",
        "cfg_override": {"mask": {"ratio": 0.10}},
    },
    "JEPA (mask=15%)": {
        "ckpt_subdir": "mask_15pct",
        "description": "Mask ratio 0.15 (default)",
        "cfg_override": {"mask": {"ratio": 0.15}},
    },
    "JEPA (mask=25%)": {
        "ckpt_subdir": "mask_25pct",
        "description": "Mask ratio 0.25",
        "cfg_override": {"mask": {"ratio": 0.25}},
    },
    "JEPA (mask=40%)": {
        "ckpt_subdir": "mask_40pct",
        "description": "Mask ratio 0.40",
        "cfg_override": {"mask": {"ratio": 0.40}},
    },
    "JEPA (EMA fixed)": {
        "ckpt_subdir": "ema_fixed",
        "description": "Fixed EMA m=0.996 (no schedule)",
        "cfg_override": {
            "training": {"ema_momentum_start": 0.996, "ema_momentum_end": 0.996}
        },
    },
    "JEPA (EMA cosine)": {
        "ckpt_subdir": "ema_cosine",
        "description": "Cosine EMA 0.990→0.9999 (default)",
        "cfg_override": {
            "training": {"ema_momentum_start": 0.990, "ema_momentum_end": 0.9999}
        },
    },
    "JEPA (1 span)": {
        "ckpt_subdir": "span_1",
        "description": "Single span per sentence",
        "cfg_override": {"mask": {"num_spans": 1}},
    },
    "JEPA (4 spans)": {
        "ckpt_subdir": "span_4",
        "description": "4 spans per sentence (default)",
        "cfg_override": {"mask": {"num_spans": 4}},
    },
    "JEPA (no pos-emb)": {
        "ckpt_subdir": "no_pos_emb",
        "description": "Predictor without positional embeddings",
        "cfg_override": {},   # Requires separate code change in predictor.py
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TABLE PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(all_results: dict):
    """
    Print a clean, aligned comparison table to stdout.

    Format:
        Model name | description | pos_sim | neg_sim | gap | accuracy | ARI | NMI | Sil.

    Numbers are aligned in columns with consistent decimal places.
    A ★ marks the best value in each column.
    """
    # Column definitions: (header, result_path, width)
    columns = [
        ("Model",       None,                              28),
        ("Description", None,                              32),
        ("Pos Sim ↑",  ("similarity", "pos_sim"),          10),
        ("Neg Sim ↓",  ("similarity", "neg_sim"),          10),
        ("Gap ↑",      ("similarity", "gap"),              10),
        ("Lin.Acc ↑",  ("linear_probe", "accuracy"),       10),
        ("ARI ↑",      ("clustering", "ari"),              10),
        ("NMI ↑",      ("clustering", "nmi"),              10),
        ("Sil. ↑",     ("clustering", "silhouette"),       10),
    ]

    # Find best value per numeric column (for ★ marker)
    numeric_cols = [(hdr, path) for hdr, path, _ in columns if path is not None]
    best = {}
    # Columns where LOWER is better
    lower_is_better = {"Neg Sim ↓"}

    for hdr, path in numeric_cols:
        vals = []
        for model_name, res in all_results.items():
            try:
                v = res[path[0]][path[1]]
                vals.append((v, model_name))
            except (KeyError, TypeError):
                pass
        if vals:
            if hdr in lower_is_better:
                best[hdr] = min(vals, key=lambda x: x[0])[1]
            else:
                best[hdr] = max(vals, key=lambda x: x[0])[1]

    # Total table width
    total_w = sum(w for _, _, w in columns) + len(columns) * 3 - 1

    sep = "─" * total_w

    print("\n")
    print("=" * total_w)
    print("  T-JEPA vs DistilBERT (MLM Baseline) — Performance Comparison")
    print("  Tasks: Semantic Similarity | Linear Probe | K-Means Clustering")
    print("  Data:  MRPC paraphrase pairs | AG News 4-class")
    print("=" * total_w)

    # Header row
    header_parts = []
    for hdr, path, w in columns:
        header_parts.append(f"{hdr:^{w}}")
    print("  " + " │ ".join(header_parts))
    print("  " + sep)

    # Section dividers
    section_breaks = {
        "DistilBERT (baseline)": "\n  ── MLM Baseline ────────────────────────────────────────────────────────────────────────────────\n",
        "T-JEPA (main)":         "\n  ── T-JEPA (main run) ───────────────────────────────────────────────────────────────────────────\n",
        "JEPA (mask=10%)":       "\n  ── Ablations: Masking Ratio ────────────────────────────────────────────────────────────────────\n",
        "JEPA (EMA fixed)":      "\n  ── Ablations: EMA Schedule ─────────────────────────────────────────────────────────────────────\n",
        "JEPA (1 span)":         "\n  ── Ablations: Span Count ───────────────────────────────────────────────────────────────────────\n",
        "JEPA (no pos-emb)":     "\n  ── Ablations: Predictor Design ─────────────────────────────────────────────────────────────────\n",
    }

    for model_name, res in all_results.items():
        # Print section break if applicable
        if model_name in section_breaks:
            print(section_breaks[model_name])

        row_parts = []
        for hdr, path, w in columns:
            if path is None:
                # Name or description column
                if hdr == "Model":
                    row_parts.append(f"{model_name:<{w}}")
                else:
                    desc = res.get("description", "")
                    row_parts.append(f"{desc:<{w}}")
            else:
                try:
                    val = res[path[0]][path[1]]
                    # Add ★ if this model is best on this metric
                    star = " ★" if best.get(hdr) == model_name else "  "
                    cell = f"{val:.4f}{star}"
                    row_parts.append(f"{cell:>{w}}")
                except (KeyError, TypeError):
                    row_parts.append(f"{'N/A':>{w}}")

        print("  " + " │ ".join(row_parts))

    print("  " + sep)
    print("\n  ★ = best in column")
    print("  ↑ = higher is better  │  ↓ = lower is better")
    print("  Gap = Pos Sim − Neg Sim (key semantic similarity metric)")
    print("=" * total_w)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T-JEPA vs Baseline Comparison")
    parser.add_argument("--ckpt",         required=True,
                        help="Path to main T-JEPA checkpoint (.pt file)")
    parser.add_argument("--config",       default="configs/base.yaml",
                        help="Config YAML path")
    parser.add_argument("--ablation-dir", default=None,
                        help="Dir containing per-ablation checkpoint subfolders. "
                             "If omitted, ablations are skipped.")
    parser.add_argument("--n-samples",    type=int, default=500,
                        help="Samples per class for AG News (default 500, total 2000)")
    parser.add_argument("--n-mrpc",       type=int, default=300,
                        help="Number of MRPC pairs per class (default 300)")
    parser.add_argument("--quick",        action="store_true",
                        help="Fast smoke test: 100 samples/class, 50 MRPC pairs")
    parser.add_argument("--save-json",    default="eval/results/comparison.json",
                        help="Save full numeric results to this JSON path")
    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.n_samples = 100
        args.n_mrpc    = 50
        print("[QUICK MODE] Using reduced dataset for fast testing")

    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\nDevice : {gpu_name}")
    print(f"Ckpt   : {args.ckpt}")
    print(f"Config : {args.config}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading evaluation data...")
    ag_texts, ag_labels, label_names = load_ag_news(n_per_class=args.n_samples)
    pos_a, pos_b, neg_a, neg_b = load_mrpc_pairs(n=args.n_mrpc)

    # Train/test split for linear probe (80/20)
    split = int(len(ag_texts) * 0.8)
    train_texts  = ag_texts[:split]
    train_labels = ag_labels[:split]
    test_texts   = ag_texts[split:]
    test_labels  = ag_labels[split:]

    # ── Helper: evaluate one model ────────────────────────────────────────────
    def evaluate_model(
        model,
        tokenizer,
        model_label: str,
        description: str,
    ) -> dict:
        """Run all 3 tasks for a single model. Returns result dict."""
        print(f"\n  Evaluating: {model_label}")

        max_len = cfg["data"]["max_length"]

        # Extract embeddings for each text set
        print("    Extracting embeddings for MRPC (similarity)...")
        embs_pos_a = extract_embeddings(pos_a, model, tokenizer, device, max_length=max_len)
        embs_pos_b = extract_embeddings(pos_b, model, tokenizer, device, max_length=max_len)
        embs_neg_a = extract_embeddings(neg_a, model, tokenizer, device, max_length=max_len)
        embs_neg_b = extract_embeddings(neg_b, model, tokenizer, device, max_length=max_len)

        print("    Extracting embeddings for AG News (probe + clustering)...")
        train_embs = extract_embeddings(train_texts, model, tokenizer, device, max_length=max_len)
        test_embs  = extract_embeddings(test_texts,  model, tokenizer, device, max_length=max_len)
        all_embs   = np.concatenate([train_embs, test_embs], axis=0)
        all_labels = np.concatenate([train_labels, test_labels], axis=0)

        # Run tasks
        print("    Running Task A: Semantic Similarity...")
        sim_result = task_semantic_similarity(embs_pos_a, embs_pos_b, embs_neg_a, embs_neg_b)

        print("    Running Task B: Linear Probe...")
        probe_result = task_linear_probe(train_embs, train_labels, test_embs, test_labels)

        print("    Running Task C: K-Means Clustering...")
        cluster_result = task_clustering(all_embs, all_labels, n_clusters=4)

        print(f"    Done. Acc={probe_result['accuracy']:.4f} | "
              f"ARI={cluster_result['ari']:.4f} | "
              f"Gap={sim_result['gap']:.4f}")

        return {
            "description":  description,
            "similarity":   sim_result,
            "linear_probe": probe_result,
            "clustering":   cluster_result,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Run evaluations
    # ══════════════════════════════════════════════════════════════════════════
    all_results = {}

    # ── 1. DistilBERT MLM Baseline ────────────────────────────────────────────
    print("\n" + "─"*60)
    print("BASELINE: DistilBERT (MLM pretrained, no additional training)")
    print("─"*60)
    db_model, db_tok = load_distilbert_baseline(device)
    all_results["DistilBERT (baseline)"] = evaluate_model(
        db_model, db_tok,
        "DistilBERT (baseline)",
        "Pretrained MLM, no SSL finetuning",
    )
    del db_model    # Free GPU memory

    # ── 2. T-JEPA Main Checkpoint ─────────────────────────────────────────────
    print("\n" + "─"*60)
    print("T-JEPA: Main trained checkpoint")
    print("─"*60)
    tjepa_model, tjepa_tok = load_tjepa_teacher(args.ckpt, cfg, device)
    all_results["T-JEPA (main)"] = evaluate_model(
        tjepa_model, tjepa_tok,
        "T-JEPA (main)",
        f"T-JEPA teacher @ {os.path.basename(args.ckpt)}",
    )

    # ── 3. Ablation Experiments ───────────────────────────────────────────────
    if args.ablation_dir:
        print("\n" + "─"*60)
        print(f"ABLATIONS: scanning {args.ablation_dir}")
        print("─"*60)

        for ablation_name, ablation_info in ABLATION_REGISTRY.items():
            ckpt_path = os.path.join(
                args.ablation_dir,
                ablation_info["ckpt_subdir"],
                "best.pt",
            )

            if not os.path.exists(ckpt_path):
                print(f"\n  [SKIP] {ablation_name} — no checkpoint at {ckpt_path}")
                continue

            print(f"\n  Ablation: {ablation_name} ({ablation_info['description']})")

            # Apply config overrides for this ablation
            abl_cfg = yaml.safe_load(yaml.dump(cfg))   # Deep copy
            for section, values in ablation_info.get("cfg_override", {}).items():
                if section in abl_cfg:
                    abl_cfg[section].update(values)
                else:
                    abl_cfg[section] = values

            try:
                abl_model, abl_tok = load_tjepa_teacher(ckpt_path, abl_cfg, device)
                all_results[ablation_name] = evaluate_model(
                    abl_model, abl_tok,
                    ablation_name,
                    ablation_info["description"],
                )
                del abl_model
            except Exception as e:
                print(f"  [ERROR] {ablation_name} failed: {e}")
    else:
        print("\n(Ablations skipped — use --ablation-dir to include them)")

    # ══════════════════════════════════════════════════════════════════════════
    # Print results table
    # ══════════════════════════════════════════════════════════════════════════
    print_results_table(all_results)

    # ══════════════════════════════════════════════════════════════════════════
    # Save to JSON
    # ══════════════════════════════════════════════════════════════════════════
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved → {args.save_json}")


if __name__ == "__main__":
    main()