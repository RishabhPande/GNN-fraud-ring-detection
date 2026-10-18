# Contributing

Contributions are welcome. This repo is deliberately dependency-free, so the
bar is: everything must run on a stock Python 3.10+ install.

## What fits here

* New GNN layers or aggregation schemes (with tests)
* Better fraud-pattern generators (new ring topologies, laundering patterns)
* Benchmarks against other from-scratch or library implementations
* Bug fixes, clearer docs, faster code

## Ground rules

1. **No new dependencies.** Pure stdlib. If your idea needs numpy, it does not
   belong in this repo (open an issue first and we can discuss).
2. **Tests for behavior changes.** `python3 -m pytest tests/ -q` must stay green.
   New layers need a hand-computed forward/backward check on a tiny graph.
3. **Deterministic.** Seed everything. No wall-clock or unseeded randomness in
   library code.
4. **Honest numbers.** If your change does not beat the baseline, say so in
   the PR. Negative results are welcome here.

## How to contribute

1. Fork the repo and create a branch from `main`.
2. Make your change with tests.
3. Run the full suite: `python3 -m pytest tests/ -q`.
4. Open a pull request describing what changed and the before/after numbers.

Good first issues are labeled `good first issue`. Questions welcome as issues.
