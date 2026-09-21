"""One frozen grouped lexical control for protected-insertion effect prediction."""
import hashlib
import json
import math
import re
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr

from src.evaluation.v17packet_obs_a1_natural_insertion import CFG, OUT, prepare, read


def tokens(text):
    words = re.findall(r"\b\w+\b", text.lower())
    return words + [a + "_" + b for a, b in zip(words, words[1:])]


def vectorize(train_text, valid_text):
    train_bags = [Counter(tokens(s)) for s in train_text]
    document_frequency = Counter()
    for bag in train_bags:
        document_frequency.update(bag.keys())
    terms = sorted((word for word, n in document_frequency.items() if n >= 2),
                   key=lambda word: (-document_frequency[word], word))[:100000]
    vocab = {word: i for i, word in enumerate(terms)}
    idf = np.asarray([math.log((1 + len(train_bags)) / (1 + document_frequency[word])) + 1
                      for word in terms], dtype=np.float32)

    def matrix(bags):
        indices, values, indptr = [], [], [0]
        for bag in bags:
            cols = [(vocab[word], (1 + math.log(count)) * idf[vocab[word]])
                    for word, count in bag.items() if word in vocab]
            norm = math.sqrt(sum(value * value for _, value in cols)) or 1
            indices.extend(column for column, _ in cols)
            values.extend(value / norm for _, value in cols)
            indptr.append(len(indices))
        return csr_matrix((np.asarray(values, dtype=np.float32), indices, indptr),
                          shape=(len(bags), len(vocab)), dtype=np.float32)

    return matrix(train_bags), matrix([Counter(tokens(s)) for s in valid_text])


def binary_auc(labels, scores):
    positives = np.flatnonzero(labels)
    negatives = np.flatnonzero(~labels)
    return float(sum((scores[p] > scores[negatives]).sum() +
                     0.5 * (scores[p] == scores[negatives]).sum() for p in positives)
                 / (len(positives) * len(negatives)))


def main():
    cfg = json.loads(CFG.read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    source, _ = prepare(cfg, tokenizer)
    labels = {r["example_id"]: r for r in read(OUT / "per_query.jsonl")}
    assert len(source) == len(labels) == 256
    texts = [
        "QUESTION: " + r["question"] + "\nBASELINE CONTEXT: " + r["baseline_context"]
        + "\nACTION CONTEXT: " + r["action_context"]
        for r in source
    ]
    y = np.asarray([labels[r["example_id"]]["delta_f1"] for r in source])
    folds = np.asarray([int(hashlib.sha256(r["example_id"].encode()).hexdigest()[:8], 16) % 4 for r in source])
    pred = np.zeros(len(source))
    for fold in range(4):
        train, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        x_train, x_valid = vectorize([texts[i] for i in train], [texts[i] for i in valid])
        intercept = float(y[train].mean())
        weights = lsqr(x_train, y[train] - intercept, damp=1.0, iter_lim=100)[0]
        pred[valid] = x_valid @ weights + intercept
    paired = []
    for i, row in enumerate(source):
        labels_row = labels[row["example_id"]]
        paired.append({"example_id": row["example_id"], "fold": int(folds[i]),
                       "predicted_delta_f1": float(pred[i]), "actual_delta_f1": float(y[i]),
                       "repair": labels_row["repair"], "break": labels_row["break"],
                       "slack": labels_row["slack"]})
    # Fold-standardized ranks avoid selecting a fold just because ridge
    # predictions have a different intercept or variance.
    ranks = np.zeros(len(source))
    for fold in range(4):
        indices = np.flatnonzero(folds == fold)
        order = indices[np.argsort(-pred[indices], kind="stable")]
        for rank, index in enumerate(order):
            ranks[index] = rank / max(1, len(order) - 1)
    decisive = np.asarray([r["repair"] or r["break"] for r in paired])
    fold_diagnostics = []
    for fold in range(4):
        local = folds == fold
        critical = local & decisive
        fold_diagnostics.append({"fold": fold,
            "queries": int(local.sum()),
            "repair_queries": int(sum(paired[i]["repair"] for i in np.flatnonzero(local))),
            "break_queries": int(sum(paired[i]["break"] for i in np.flatnonzero(local))),
            "spearman": float(spearmanr(pred[local], y[local]).statistic),
            "repair_vs_break_auc": binary_auc(
                np.asarray([r["repair"] for r in paired])[critical], pred[critical])})
    summary = {"protocol": "OBS-A2_GROUPED_FULL_PAIRED_TEXT_TFIDF_RIDGE_CONTROL",
               "queries": len(paired), "fold_sizes": [int(sum(folds == k)) for k in range(4)],
               "pooled_spearman_predicted_vs_actual_delta_f1": float(spearmanr(pred, y).statistic),
               "pooled_repair_vs_break_auc_on_decisive_cases": binary_auc(
                   np.asarray([r["repair"] for r in paired])[decisive], pred[decisive]),
               "fold_diagnostics": fold_diagnostics,
               "budgets": {}, "sealed_sets_read": False,
               "interpretation": "A cheap lexical control on 256 train-side queries, not a strong observability ceiling or deployable model."}
    for fraction in (0.05, 0.10):
        chosen = sorted(range(len(paired)), key=lambda i: (ranks[i], paired[i]["example_id"]))[:round(len(paired) * fraction)]
        summary["budgets"][str(fraction)] = {"interventions": len(chosen),
            "repairs": sum(paired[i]["repair"] for i in chosen),
            "breaks": sum(paired[i]["break"] for i in chosen),
            "mean_extra_context_tokens_all_queries": sum(paired[i]["slack"] for i in chosen) / len(paired),
            "random_expected_repairs": len(chosen) * sum(r["repair"] for r in paired) / len(paired),
            "random_expected_breaks": len(chosen) * sum(r["break"] for r in paired) / len(paired)}
    (OUT / "paired_text_control_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "paired_text_control_oof.jsonl").open("w") as handle:
        for row in paired:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
