"""Edge classification: which flows move the money, not just which accounts
look suspicious.

A GraphSAGE encoder is trained on node labels first and then frozen. An
MLP edge head scores every edge from the pair of endpoint embeddings
(phi = [h_u, h_v, h_u*h_v, |h_u-h_v|]). Edges are labeled 1 when both
endpoints are ring members (a ring-internal flow), 0 otherwise.

Honest headline: the head is a probe of what the node encoder already
learned, so edges are an easier task than nodes here -- the dense pocket
is the whole story again. The bridge edges (ring member to benign
account) are the genuinely hard negatives: they look structurally like
background but carry one suspicious endpoint.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn_fraud import (
    GraphGenerator,
    best_threshold_f1,
    edge_labels,
    edge_train_val_split,
    evaluate_all,
    score_edges,
    train_edge_cls,
    train_sage,
    train_val_split,
)

gen = GraphGenerator(seed=7)
g = gen.generate(n=200, ring_size=18)
ring = set(gen.ring_nodes(g))
labels = edge_labels(g)
edges = sorted(labels)
n_pos = sum(labels.values())
print(f"generated {g.n} accounts, {len(edges)} edges, "
      f"{n_pos} ring-internal ({len(edges) - n_pos} background)")

print("\ntraining GraphSAGE node encoder (frozen afterwards)...")
train_ids, val_ids = train_val_split(g.labels, val_frac=0.25, seed=7)
encoder, _ = train_sage(g, train_ids, val_ids, hidden_dim=16,
                        k1=5, k2=5, lr=0.1, batch_size=64,
                        epochs=60, patience=12, seed=7)

print("training edge head on frozen embeddings...")
head, emb, _ = train_edge_cls(g, encoder, hidden_dim=16, lr=0.1,
                              batch_size=64, epochs=60, patience=12, seed=7)

_, val_edges = edge_train_val_split(edges, labels, seed=7)
scores = score_edges(head, emb, val_edges)
y_true = [labels[e] for e in val_edges]
y_score = [scores[e] for e in val_edges]

print(f"\nedge metrics on {len(val_edges)} held-out edges "
      "(F1-tuned threshold):")
print(f"  {'thr':>6}{'precision':>10}{'recall':>8}{'F1':>8}{'AUC':>8}")
thr, m = best_threshold_f1(y_true, y_score)
auc = evaluate_all(y_true, y_score)["auc"]
print(f"  {thr:>6.2f}{m['precision']:>10.3f}{m['recall']:>8.3f}"
      f"{m['f1']:>8.3f}{auc:>8.3f}")

# Honest nuance: bridge edges (one suspicious endpoint, background label)
# are the hard cases. Score every edge type with the trained head.
all_scores = score_edges(head, emb, edges)
ring_int = [all_scores[e] for e in edges if labels[e] == 1]
bridges = [all_scores[e] for e in edges
           if labels[e] == 0 and ((e[0] in ring) != (e[1] in ring))]
benign = [all_scores[e] for e in edges
          if labels[e] == 0 and not ((e[0] in ring) != (e[1] in ring))]
mean = lambda xs: sum(xs) / len(xs)
print("\nmean head score by edge type (all edges):")
print(f"  ring-internal ({len(ring_int):>3}): {mean(ring_int):.3f}")
print(f"  bridge        ({len(bridges):>3}): {mean(bridges):.3f}")
print(f"  benign-benign ({len(benign):>3}): {mean(benign):.3f}")
print("\nTakeaway: ring-internal flows separate cleanly, but bridges sit "
      "halfway -- one suspicious endpoint leaks signal. A production "
      "edge model would add flow features (amount, direction, timing) "
      "instead of relying on structure alone.")
