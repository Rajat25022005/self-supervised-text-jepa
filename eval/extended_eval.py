import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import average_precision_score


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — model loading + embedding extraction
# ══════════════════════════════════════════════════════════════════════════════

def load_tjepa(ckpt_path: str, cfg: dict, device: torch.device, layer: str = "projected"):
    """
    Load T-JEPA teacher encoder.

    layer="projected" → 256-dim JEPA-trained latents  (use for all T-JEPA tasks)
    layer="raw"       → 768-dim pre-projection         (use for fair baseline compare)
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.models.encoder import TextEncoder

    ckpt = torch.load(ckpt_path, map_location=device)
    enc  = TextEncoder(cfg["model"]["encoder_name"], cfg["model"]["proj_dim"])
    enc.load_state_dict(ckpt["teacher_state_dict"])
    enc.eval().to(device)

    model = enc if layer == "projected" else enc.encoder
    tok   = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    dim   = cfg["model"]["proj_dim"] if layer == "projected" else cfg["model"]["hidden_size"]

    print(f"  T-JEPA teacher loaded ({layer}, {dim}-dim)")
    return model, tok, dim


def load_distilbert(device: torch.device):
    """Load vanilla DistilBERT baseline."""
    name = "distilbert-base-uncased"
    tok  = AutoTokenizer.from_pretrained(name)
    m    = AutoModel.from_pretrained(name).eval().to(device)
    print(f"  DistilBERT baseline loaded (768-dim)")
    return m, tok, 768


@torch.no_grad()
def get_embeddings(
    texts: list,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 128,
    max_length: int = 128,
    pool: str = "cls",
) -> np.ndarray:
    """Extract embeddings — handles both HF models and TextEncoder."""
    model.eval()
    all_embs = []

    for i in range(0, len(texts), batch_size):
        toks = tokenizer(
            texts[i : i + batch_size],
            padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        ids  = toks["input_ids"].to(device)
        mask = toks["attention_mask"].to(device)

        out    = model(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out

        if pool == "cls":
            emb = hidden[:, 0, :]
        else:
            m2  = mask.unsqueeze(-1).float()
            emb = (hidden * m2).sum(1) / m2.sum(1).clamp(min=1e-9)

        all_embs.append(emb.cpu().float().numpy())

    return np.concatenate(all_embs, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_ag_news(n_per_class: int = 500):
    """AG News 4-class — standard SSL classification benchmark."""
    from datasets import load_dataset

    print("  Loading AG News...")
    ds          = load_dataset("ag_news", split="test")
    label_names = ["World", "Sports", "Business", "Sci/Tech"]

    texts, labels = [], []
    for cls in range(4):
        pool = [(x["text"], x["label"]) for x in ds if x["label"] == cls][:n_per_class]
        for t, l in pool:
            texts.append(t)
            labels.append(l)

    idx    = np.random.permutation(len(texts))
    texts  = [texts[i]  for i in idx]
    labels = np.array([labels[i] for i in idx])

    print(f"    {len(texts)} samples loaded")
    return texts, labels, label_names


def load_mrpc(n: int = 300):
    """MRPC paraphrase pairs — for predictive consistency."""
    from datasets import load_dataset

    print("  Loading MRPC...")
    ds = load_dataset("glue", "mrpc", split="test")
    pairs = [(x["sentence1"], x["sentence2"], x["label"]) for x in ds][:n]
    s1 = [p[0] for p in pairs]
    s2 = [p[1] for p in pairs]
    lb = np.array([p[2] for p in pairs])

    print(f"    {len(pairs)} pairs loaded")
    return s1, s2, lb


# ══════════════════════════════════════════════════════════════════════════════
# A. RETRIEVAL — Recall@K and mAP
# ══════════════════════════════════════════════════════════════════════════════

def task_retrieval(embs: np.ndarray, labels: np.ndarray, ks: list = [1, 5, 10]) -> dict:
    """
    Recall@K and Mean Average Precision (mAP).

    Setup:
        For each query embedding, rank ALL other embeddings by cosine similarity.
        A "relevant" document = same class label as the query.

    Recall@K:
        Fraction of queries where at least one relevant item
        appears in the top-K results.
        Recall@1 = how often the nearest neighbour is the right class.
        Recall@10 = how often a correct match appears in top 10.

    mAP (Mean Average Precision):
        Average precision is computed per query — it rewards finding
        relevant items early in the ranked list.
        mAP averages this across all queries.
        Higher mAP = relevant items are consistently ranked near the top.

    Interpretation:
        Recall@1 > 0.7  : very good nearest-neighbour retrieval
        mAP > 0.6       : strong ranking quality
        Both near 1.0   : embeddings are perfectly class-separated

    Args:
        embs   : (N, D) normalized embeddings
        labels : (N,) integer class labels
        ks     : list of K values for Recall@K

    Returns:
        dict with recall@1, recall@5, recall@10, mAP
    """
    print("    Computing Recall@K and mAP...")

    normed    = normalize(embs, norm="l2")
    sim_matrix = normed @ normed.T           # (N, N) cosine similarity

    N       = len(labels)
    max_k   = max(ks)
    recalls = {k: [] for k in ks}
    aps     = []

    for i in range(N):
        # Get similarity scores excluding self (diagonal)
        sims = sim_matrix[i].copy()
        sims[i] = -2.0                        # exclude self

        # Sorted indices from most to least similar
        ranked = np.argsort(-sims)            # (N-1,)

        # Ground truth: which others share the same label
        relevant = (labels[ranked] == labels[i])

        # Recall@K
        for k in ks:
            recalls[k].append(int(relevant[:k].any()))

        # Average Precision for this query
        # AP = (sum of precision@k for each relevant hit) / total relevant
        n_relevant = relevant.sum()
        if n_relevant == 0:
            aps.append(0.0)
            continue

        hits, ap_sum = 0, 0.0
        for rank, is_rel in enumerate(relevant, start=1):
            if is_rel:
                hits   += 1
                ap_sum += hits / rank        # precision at this rank

        aps.append(ap_sum / n_relevant)

    result = {f"recall@{k}": float(np.mean(recalls[k])) for k in ks}
    result["mAP"] = float(np.mean(aps))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# B. FEW-SHOT — 1-shot and 5-shot classification
# ══════════════════════════════════════════════════════════════════════════════

def task_few_shot(
    embs: np.ndarray,
    labels: np.ndarray,
    shots: list = [1, 5],
    n_episodes: int = 200,
    n_way: int = 4,
) -> dict:
    """
    N-way K-shot classification with a frozen encoder + nearest centroid head.

    Protocol (standard in SSL papers):
        1. Sample N classes randomly from the label set
        2. For each class, sample K "support" embeddings (the few-shot examples)
        3. Compute the class centroid = mean of K support embeddings
        4. Classify remaining "query" embeddings by nearest centroid (cosine)
        5. Repeat for n_episodes episodes, average accuracy

    Why nearest centroid?
        It's the lightest possible head — zero trainable parameters.
        All learning must come from the encoder. This strictly measures
        representation quality, not head capacity.

    Why N-way?
        We use all available classes (n_way = number of unique labels).
        This avoids the artificial difficulty of sampling random class subsets
        and gives more stable estimates.

    Interpretation:
        1-shot accuracy > 0.6 : encoder captures class identity from 1 example
        5-shot accuracy > 0.75: strong few-shot learner
        5-shot ≈ linear probe  : representations are already linearly separable

    Args:
        embs      : (N, D) embeddings
        labels    : (N,) integer class labels
        shots     : list of K values (e.g. [1, 5])
        n_episodes: number of random episodes to average over
        n_way     : number of classes (use all unique labels)

    Returns:
        dict with 1shot_acc, 5shot_acc, etc.
    """
    print(f"    Computing {shots}-shot classification ({n_episodes} episodes)...")

    normed    = normalize(embs, norm="l2")
    classes   = np.unique(labels)
    n_way     = min(n_way, len(classes))
    results   = {k: [] for k in shots}

    # Build per-class index lists for fast sampling
    class_idx = {c: np.where(labels == c)[0].tolist() for c in classes}

    for _ in range(n_episodes):
        # Sample n_way classes for this episode
        ep_classes = random.sample(list(classes), n_way)

        for k in shots:
            support_embs   = []
            support_labels = []
            query_embs     = []
            query_labels   = []

            for ep_cls_idx, cls in enumerate(ep_classes):
                idx = class_idx[cls]

                # Need at least k+1 examples (k support + 1 query)
                if len(idx) < k + 1:
                    continue

                sampled = random.sample(idx, k + 1)
                support = sampled[:k]
                query   = sampled[k:]

                support_embs.extend(normed[support])
                support_labels.extend([ep_cls_idx] * k)
                query_embs.extend(normed[query])
                query_labels.extend([ep_cls_idx] * len(query))

            if not support_embs or not query_embs:
                continue

            support_embs = np.array(support_embs)     # (n_way*k, D)
            query_embs   = np.array(query_embs)       # (n_queries, D)
            support_labels = np.array(support_labels)
            query_labels   = np.array(query_labels)

            # Compute class centroids from support set
            centroids = np.array([
                support_embs[support_labels == i].mean(axis=0)
                for i in range(n_way)
            ])                                         # (n_way, D)

            # Normalize centroids
            centroids = normalize(centroids, norm="l2")

            # Nearest centroid classification (cosine)
            sims  = query_embs @ centroids.T           # (n_queries, n_way)
            preds = sims.argmax(axis=1)
            acc   = (preds == query_labels).mean()
            results[k].append(float(acc))

    return {f"{k}shot_acc": float(np.mean(v)) if v else 0.0
            for k, v in results.items()}


# ══════════════════════════════════════════════════════════════════════════════
# C. PREDICTIVE CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

def task_predictive_consistency(
    model,
    tokenizer,
    s1_texts: list,
    s2_texts: list,
    pair_labels: np.ndarray,
    device: torch.device,
    cfg: dict,
) -> dict:
    """
    Predictive Consistency: how well the encoder predicts related content.

    Two sub-metrics:

    1. Context→Future Embedding Error
    ──────────────────────────────────
    Given a sentence pair (s1, s2) where s2 is semantically related to s1
    (paraphrase, label=1) or unrelated (label=0):

        error = ||embed(s1) - embed(s2)||²  (L2 distance in embedding space)

    A good JEPA encoder should have LOW error for related pairs and
    HIGH error for unrelated pairs. The ratio (unrelated_error / related_error)
    measures predictive discriminability.

    2. Masked-Span Reconstruction Error (Embedding Space)
    ──────────────────────────────────────────────────────
    Given a sentence, mask a random span and compare:
        embed(masked_sentence) vs embed(original_sentence)

    This directly measures what JEPA was trained to do — predicting masked
    content in latent space. A well-trained JEPA encoder should produce
    embeddings for masked text that are close to the clean text embeddings.

    Lower reconstruction error = encoder has learned to predict masked content.
    This is the metric MOST specific to the JEPA objective.

    Args:
        model       : encoder model
        tokenizer   : matching tokenizer
        s1_texts    : list of first sentences
        s2_texts    : list of second sentences (paired with s1)
        pair_labels : 1 = paraphrase, 0 = unrelated
        device      : torch device
        cfg         : config dict

    Returns:
        dict with related_error, unrelated_error, discriminability_ratio,
              masked_reconstruction_error, clean_reconstruction_error
    """
    print("    Computing predictive consistency metrics...")

    max_len = cfg["data"]["max_length"]
    bs      = cfg["eval"]["batch_size"]

    # ── Sub-metric 1: Context → Future embedding error ──────────────────────
    embs_s1 = get_embeddings(s1_texts, model, tokenizer, device,
                             batch_size=bs, max_length=max_len)
    embs_s2 = get_embeddings(s2_texts, model, tokenizer, device,
                             batch_size=bs, max_length=max_len)

    embs_s1 = normalize(embs_s1, norm="l2")
    embs_s2 = normalize(embs_s2, norm="l2")

    # L2 distance in normalized space = sqrt(2 - 2*cosine_sim)
    l2_sq = ((embs_s1 - embs_s2) ** 2).sum(axis=1)     # (N,)

    pos_mask = pair_labels == 1
    neg_mask = pair_labels == 0

    related_error   = float(l2_sq[pos_mask].mean())     # should be low
    unrelated_error = float(l2_sq[neg_mask].mean())     # should be high

    # Discriminability: ratio of unrelated to related error
    # Higher = better separation between related and unrelated pairs
    discriminability = unrelated_error / (related_error + 1e-9)

    # ── Sub-metric 2: Masked-span reconstruction error ───────────────────────
    # Apply BERT-style masking to a subset of sentences and measure
    # how close the masked embeddings are to the clean embeddings
    sample_texts = s1_texts[:100]   # use first 100 for speed

    # Get clean embeddings
    clean_embs = get_embeddings(sample_texts, model, tokenizer, device,
                                batch_size=bs, max_length=max_len)
    clean_embs = normalize(clean_embs, norm="l2")

    # Tokenize and manually mask a span
    toks = tokenizer(
        sample_texts,
        padding=True, truncation=True,
        max_length=max_len, return_tensors="pt"
    )

    masked_ids = toks["input_ids"].clone()
    B, T       = masked_ids.shape
    mask_token = tokenizer.mask_token_id

    for i in range(B):
        # Mask a random span of 5–10 tokens (skip CLS=0, SEP=T-1)
        span_len = random.randint(5, min(10, T - 2))
        start    = random.randint(1, max(2, T - span_len - 1))
        masked_ids[i, start : start + span_len] = mask_token

    # Get masked embeddings
    with torch.no_grad():
        out = model(
            input_ids=masked_ids.to(device),
            attention_mask=toks["attention_mask"].to(device)
        )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        masked_embs_t = hidden[:, 0, :].cpu().float().numpy()

    masked_embs = normalize(masked_embs_t, norm="l2")

    # Reconstruction error = L2 distance between masked and clean embeddings
    recon_error = float(((clean_embs - masked_embs) ** 2).sum(axis=1).mean())

    # Baseline: compare two unrelated sentences (upper bound on error)
    shuffled    = clean_embs[np.random.permutation(len(clean_embs))]
    random_error = float(((clean_embs - shuffled) ** 2).sum(axis=1).mean())

    # Reconstruction quality: how close is masked to clean vs random baseline?
    # 0.0 = perfect reconstruction, 1.0 = as bad as random
    recon_quality = 1.0 - (recon_error / (random_error + 1e-9))

    return {
        "related_error":         related_error,
        "unrelated_error":       unrelated_error,
        "discriminability_ratio":discriminability,    # KEY: higher = better
        "masked_recon_error":    recon_error,         # KEY: lower = better
        "random_baseline_error": random_error,
        "recon_quality":         recon_quality,       # KEY: higher = better (0-1)
    }


# ══════════════════════════════════════════════════════════════════════════════
# D. MLP PROBE — 1-hidden-layer nonlinear probe
# ══════════════════════════════════════════════════════════════════════════════

class MLPProbe(nn.Module):
    """
    Shallow 1-hidden-layer MLP classification head.

    Architecture: Linear → ReLU → Dropout → Linear → Softmax

    Why MLP probe vs linear probe?
        Linear probe measures linear separability — can a hyperplane separate
        the classes in embedding space?

        MLP probe measures general separability — can any smooth nonlinear
        boundary separate the classes?

        If MLP >> Linear: embeddings have structure but it's nonlinear.
        If MLP ≈ Linear: embeddings are already linearly separable (ideal for SSL).
        If both are low: representations don't encode class information at all.

        The gap (MLP_acc - Linear_acc) tells you how much "nonlinear leftover"
        structure exists that linear probing misses.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def task_mlp_probe(
    train_embs:   np.ndarray,
    train_labels: np.ndarray,
    test_embs:    np.ndarray,
    test_labels:  np.ndarray,
    emb_dim:      int,
    hidden_dim:   int = 256,
    n_epochs:     int = 50,
    lr:           float = 1e-3,
    device: torch.device = None,
) -> dict:
    """
    Train a 1-hidden-layer MLP probe on frozen embeddings.

    Training:
        - Adam optimizer, cross-entropy loss
        - 50 epochs (fast — embeddings are pre-computed, no backprop through encoder)
        - Batch gradient descent on all training embeddings

    Args:
        train_embs/labels : training split
        test_embs/labels  : test split
        emb_dim           : input dimension (256 for projected, 768 for raw)
        hidden_dim        : MLP hidden layer width (default 256)
        n_epochs          : training epochs (50 is enough for a probe)
        lr                : Adam learning rate
        device            : torch device

    Returns:
        dict with accuracy, final_train_loss, gap_vs_linear
    """
    print(f"    Training MLP probe ({emb_dim}→{hidden_dim}→{len(np.unique(train_labels))})...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_classes  = len(np.unique(train_labels))
    le         = LabelEncoder()
    tr_l_enc   = le.fit_transform(train_labels)
    te_l_enc   = le.transform(test_labels)

    # Normalize embeddings
    tr_norm = normalize(train_embs, norm="l2")
    te_norm = normalize(test_embs,  norm="l2")

    # Convert to tensors
    X_tr = torch.tensor(tr_norm,  dtype=torch.float32).to(device)
    y_tr = torch.tensor(tr_l_enc, dtype=torch.long).to(device)
    X_te = torch.tensor(te_norm,  dtype=torch.float32).to(device)
    y_te = torch.tensor(te_l_enc, dtype=torch.long).to(device)

    # Build and train probe
    probe = MLPProbe(emb_dim, hidden_dim, n_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    probe.train()
    final_loss = 0.0
    for epoch in range(n_epochs):
        opt.zero_grad()
        logits = probe(X_tr)
        loss   = F.cross_entropy(logits, y_tr)
        loss.backward()
        opt.step()
        final_loss = loss.item()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        preds = probe(X_te).argmax(dim=1).cpu().numpy()

    accuracy = float((preds == te_l_enc).mean())

    return {
        "accuracy":        accuracy,
        "final_train_loss":float(final_loss),
        "hidden_dim":      hidden_dim,
        "n_epochs":        n_epochs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TABLE PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def print_extended_table(results: dict):
    """Print all extended metrics in grouped, readable tables."""

    total_w = 100
    print("\n" + "=" * total_w)
    print("  T-JEPA EXTENDED EVALUATION RESULTS")
    print("=" * total_w)

    sections = [
        ("A. RETRIEVAL", [
            ("Recall@1 ↑",         "retrieval", "recall@1"),
            ("Recall@5 ↑",         "retrieval", "recall@5"),
            ("Recall@10 ↑",        "retrieval", "recall@10"),
            ("mAP ↑",              "retrieval", "mAP"),
        ]),
        ("B. FEW-SHOT CLASSIFICATION", [
            ("1-Shot Acc ↑",       "few_shot",  "1shot_acc"),
            ("5-Shot Acc ↑",       "few_shot",  "5shot_acc"),
        ]),
        ("C. PREDICTIVE CONSISTENCY", [
            ("Related Error ↓",    "consistency", "related_error"),
            ("Unrelated Error ↑",  "consistency", "unrelated_error"),
            ("Discriminability ↑", "consistency", "discriminability_ratio"),
            ("Masked Recon Err ↓", "consistency", "masked_recon_error"),
            ("Recon Quality ↑",    "consistency", "recon_quality"),
        ]),
        ("D. MLP PROBE", [
            ("MLP Accuracy ↑",     "mlp_probe", "accuracy"),
            ("Train Loss ↓",       "mlp_probe", "final_train_loss"),
        ]),
    ]

    lower_is_better_metrics = {
        "Related Error ↓", "Masked Recon Err ↓", "Train Loss ↓"
    }
    higher_is_better_for_unrelated = {"Unrelated Error ↑"}

    model_names = list(results.keys())
    col_w = 22

    for section_name, metrics in sections:
        print(f"\n  {'─'*total_w}")
        print(f"  {section_name}")
        print(f"  {'─'*total_w}")

        # Header
        header = f"  {'Metric':<28}" + "".join(f"{n:>{col_w}}" for n in model_names)
        print(header)
        print(f"  {'─'*total_w}")

        for metric_label, task_key, metric_key in metrics:
            row = f"  {metric_label:<28}"

            # Find best value for ★
            vals = []
            for name in model_names:
                try:
                    v = results[name][task_key][metric_key]
                    vals.append((v, name))
                except (KeyError, TypeError):
                    pass

            if vals:
                if metric_label in lower_is_better_metrics:
                    best_name = min(vals, key=lambda x: x[0])[1]
                else:
                    best_name = max(vals, key=lambda x: x[0])[1]
            else:
                best_name = None

            for name in model_names:
                try:
                    val  = results[name][task_key][metric_key]
                    star = " ★" if name == best_name else "  "
                    cell = f"{val:.4f}{star}"
                    row += f"{cell:>{col_w}}"
                except (KeyError, TypeError):
                    row += f"{'N/A':>{col_w}}"

            print(row)

    print(f"\n  {'─'*total_w}")
    print("  ★ = best in row  |  ↑ = higher better  |  ↓ = lower better")
    print(f"  Discriminability = unrelated_error / related_error  (higher = better separation)")
    print(f"  Recon Quality    = 1 - (masked_error / random_baseline)  (0=worst, 1=perfect)")
    print("=" * total_w)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="T-JEPA Extended Evaluation")
    parser.add_argument("--ckpt",      required=True,  help="Checkpoint path (.pt)")
    parser.add_argument("--config",    default="configs/eval.yaml")
    parser.add_argument("--baseline",  action="store_true",
                        help="Also evaluate DistilBERT baseline")
    parser.add_argument("--quick",     action="store_true",
                        help="Reduced dataset for fast testing (~3 min)")
    parser.add_argument("--save-json", default="eval/results/extended.json")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_per_class = 100  if args.quick else cfg["eval"]["linear_probe"]["n_per_class"]
    n_pairs     = 100  if args.quick else cfg["eval"]["similarity"]["n_pairs"]
    n_episodes  = 50   if args.quick else 200
    n_epochs    = 20   if args.quick else 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu    = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n{'='*60}")
    print(f"  T-JEPA Extended Evaluation")
    print(f"  Device : {gpu}")
    print(f"  Ckpt   : {args.ckpt}")
    print(f"  Quick  : {args.quick}")
    print(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    ag_texts, ag_labels, _ = load_ag_news(n_per_class=n_per_class)
    s1, s2, pair_labels    = load_mrpc(n=n_pairs)

    split       = int(len(ag_texts) * 0.8)
    train_texts = ag_texts[:split];  train_labels = ag_labels[:split]
    test_texts  = ag_texts[split:];  test_labels  = ag_labels[split:]

    max_len = cfg["data"]["max_length"]
    bs      = cfg["eval"]["batch_size"]

    # ── Evaluate one model — run all 4 tasks ──────────────────────────────────
    def evaluate(model, tokenizer, name: str, emb_dim: int) -> dict:
        print(f"\n{'─'*60}")
        print(f"  Evaluating: {name} ({emb_dim}-dim)")
        print(f"{'─'*60}")

        # Extract embeddings once, reuse across tasks
        print("  Extracting AG News embeddings...")
        tr_embs = get_embeddings(train_texts, model, tokenizer, device, bs, max_len)
        te_embs = get_embeddings(test_texts,  model, tokenizer, device, bs, max_len)
        all_embs = np.concatenate([tr_embs, te_embs])
        all_lbls = np.concatenate([train_labels, test_labels])

        # A. Retrieval
        print("\n  [A] Retrieval...")
        retrieval = task_retrieval(all_embs, all_lbls)
        print(f"      Recall@1={retrieval['recall@1']:.4f} | "
              f"Recall@5={retrieval['recall@5']:.4f} | "
              f"mAP={retrieval['mAP']:.4f}")

        # B. Few-shot
        print("\n  [B] Few-shot classification...")
        few_shot = task_few_shot(all_embs, all_lbls,
                                 shots=[1, 5], n_episodes=n_episodes)
        print(f"      1-shot={few_shot['1shot_acc']:.4f} | "
              f"5-shot={few_shot['5shot_acc']:.4f}")

        # C. Predictive consistency
        print("\n  [C] Predictive consistency...")
        consistency = task_predictive_consistency(
            model, tokenizer, s1, s2, pair_labels, device, cfg)
        print(f"      Discriminability={consistency['discriminability_ratio']:.4f} | "
              f"ReconQuality={consistency['recon_quality']:.4f}")

        # D. MLP probe
        print("\n  [D] MLP probe...")
        mlp = task_mlp_probe(tr_embs, train_labels, te_embs, test_labels,
                             emb_dim=emb_dim, hidden_dim=256,
                             n_epochs=n_epochs, device=device)
        print(f"      MLP Accuracy={mlp['accuracy']:.4f}")

        return {
            "retrieval":   retrieval,
            "few_shot":    few_shot,
            "consistency": consistency,
            "mlp_probe":   mlp,
        }

    all_results = {}

    # ── T-JEPA (projected) — main result ──────────────────────────────────────
    model, tok, dim = load_tjepa(args.ckpt, cfg, device, layer="projected")
    all_results["T-JEPA (projected)"] = evaluate(model, tok, "T-JEPA projected", dim)
    del model

    # ── T-JEPA (raw) — pre-projection baseline ────────────────────────────────
    model, tok, dim = load_tjepa(args.ckpt, cfg, device, layer="raw")
    all_results["T-JEPA (raw)"] = evaluate(model, tok, "T-JEPA raw", dim)
    del model

    # ── DistilBERT baseline (optional) ────────────────────────────────────────
    if args.baseline:
        model, tok, dim = load_distilbert(device)
        all_results["DistilBERT"] = evaluate(model, tok, "DistilBERT baseline", dim)
        del model

    # ── Print table ───────────────────────────────────────────────────────────
    print_extended_table(all_results)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
    with open(args.save_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {args.save_json}")


if __name__ == "__main__":
    main()