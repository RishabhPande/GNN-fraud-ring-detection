---
title: 'GNN Fraud Ring Detection: Graph Neural Networks from Scratch for Fraud Ring Detection'
tags:
  - Python
  - graph neural networks
  - fraud detection
  - GraphSAGE
  - graph attention networks
  - anomaly detection
authors:
  - name: Rishabh Pande
    orcid: 0009-0004-1901-7109
    affiliation: "1"
affiliations:
  - name: Independent Researcher
    index: 1
date: 27 October 2026
bibliography: paper.bib
---

# Summary

GNN Fraud Ring Detection is a dependency-free Python library that implements graph neural networks from scratch, using only the standard library, for detecting fraud rings. It provides GraphSAGE [@hamilton2017inductive] with mean, max-pool, and LSTM aggregation, graph convolutional networks [@kipf2017semi], graph attention networks [@velickovic2018graph] including multi-head attention, graph isomorphism networks [@xu2019powerful], relational GraphSAGE for heterogeneous graphs of accounts, devices, and merchants [@schlichtkrull2018modeling], and recency-weighted temporal GraphSAGE. Every model is implemented end to end: forward pass, backpropagation through message-passing layers, neighbor sampling, and mini-batch stochastic gradient descent. The library ships with synthetic fraud-pattern generators (dense fraud rings, money-mule chains, and smurfing/structuring networks) whose node features are near-camouflaged, so the signal lives in the graph wiring rather than in individual accounts. It also provides evaluation protocols for inductive prediction on unseen graphs and for edge classification, a Colab quickstart, and 146 automated tests. All data is synthetic; no real people, transactions, or personally identifiable information are involved.

# Statement of need

A fraud ring can look perfectly ordinary one account at a time: normal ages, normal balances, normal transaction sizes. What gives a ring away is the wiring -- a dense pocket of accounts transacting with each other far more than chance allows, a chain of pass-through mules, or a fan-out/fan-in structuring pattern. Feature-only models miss this structure by construction. Existing graph-learning frameworks are powerful but heavyweight: they pull in large tensor libraries and assume GPU-backed workflows, which raises the barrier for researchers and practitioners who want to study fraud patterns, teach message passing, or benchmark detectors without touching sensitive data. GNN Fraud Ring Detection fills that gap with readable, from-scratch implementations and generators that let anyone build, test, and compare detectors on realistic fraud structure without real transactions. On identical protocols, the graph signal dominates feature-only baselines: on the smurfing pattern, GraphSAGE reaches AUC 0.933 against 0.672 for a feature-only multilayer perceptron, and on the dense-ring pattern, max-pool GraphSAGE reaches AUC 1.000 against 0.656. Because every architecture trains and evaluates under the same protocol, the library also serves as a controlled testbed for architectural comparisons -- for example, showing where sum aggregation preserves neighborhood counts that mean aggregation erases, and where degree normalization dilutes the ring signal.

# Acknowledgements

The author thanks the open-source graph-learning community whose published architectures this library re-implements from scratch for study and benchmarking.

# References
