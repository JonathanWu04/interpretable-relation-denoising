# Interpretable Relation Denoising

A CPU-only study of how to remove unreliable relation labels **while retaining useful supervision**. The project grew from an NYU NLP course investigation into a study of learned rule weights, contextual features, and data-retention objectives. Team contributions and implementation assistance are documented in [Provenance](docs/PROVENANCE.md).

## Result at a glance

A sparse logistic filter combines interpretable linguistic rules with **entity-masked local words and bigrams conditioned on the candidate relation**. Compared with learning only the original rule weights, it improves the measured quality/retention trade-off:

| Evaluation | Method | Clean examples retained | Noise removed | Retained-set purity |
|---|---|---:|---:|---:|
| Development: 5 outer sentence-grouped folds | Rule-feature LR | 90.94% | 19.58% | 77.42% |
| Development: 5 outer sentence-grouped folds | Rules + context + relation LR | **89.93%** | **31.84%** | **80.00%** |
| Diagnostic test, no development-sentence overlap | Rule-feature LR | 90.36% | 21.86% | 77.92% |
| Diagnostic test, no development-sentence overlap | Rules + context + relation LR | **92.37%** | **30.44%** | **80.20%** |

Thresholds target 90% clean retention on inner development predictions, separately for NA/non-NA. Actual retention can differ. The new filter retains **60 more clean examples and removes 84 more noisy examples** on the 3,966-example main diagnostic under frozen policies.

These are **noise-detection** results, not relation-classification F1 or a comparison with BERT/ARNOR. The test corpus was examined in previous project iterations, so it is diagnostic rather than a pristine holdout. On the separate noise-focused diagnostic the context model **does not outperform** rule-feature LR. The full report includes that result and the near-tie of a context-only ablation.

## Read the work

- [Research paper (PDF)](report/technical_report.pdf), [editable ACL LaTeX source](report/latex/main.tex), and [detailed experiment notes](report/technical_report.md)
- [Frozen experiment protocol](configs/protocol.json)
- [Complete metrics](results/20261114/results.json), [paired bootstrap intervals](results/20261114/oof_paired_bootstrap.json), and [model JSON](results/20261114/models/context_lr.json)
- [Reproduction instructions](docs/REPRODUCIBILITY.md)
- [Original contributions, history, and rights](docs/PROVENANCE.md)

## What changed

1. **Objective:** measure clean retention, noise recall, and retained-set purity together; provide Retain90, Retain95, and Recall90 policies.
2. **Model:** compare hand-weighted rules, learned rule weights, relation priors, relation-conditioned rules, local context, and their combination.
3. **Protocol:** train-only vocabulary/scaling, nested sentence-grouped selection, development-only frozen thresholds, exact duplicate handling, overlap diagnostics, and paired sentence-bootstrap uncertainty.
4. **Usability:** a small JSON model, per-mention keep/drop/abstain output, data checksums, unit tests, and independently verifiable result tables.

Training and inference use CPU only. The released context model is about 4.1 MB as readable JSON. Locally, feature extraction, scoring, and decisions for 4,050 valid examples took a median of 0.255 seconds, excluding file IO; this is not a hardware-matched neural-model speedup claim.

## Quick start

Use Python 3.12 for the locked environment.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python scripts/download_data.py --out data/arnor
python -m relden.experiment --data-dir data/arnor --out results/reproduction
python scripts/verify_release.py results/reproduction
```

Audit the bundled results without downloading the dataset:

```sh
python scripts/verify_release.py results/20261114
```

Score new ARNOR-format JSONL:

```sh
relden --model results/20261114/models/context_lr.json \
  --input data/arnor/test.json --output decisions.jsonl --policy retain90
```

Ambiguous or missing spans produce `abstain`. Gold noise labels are not used by inference. All output paths must be new so old results are not overwritten.

## Repository layout

```text
src/relden/       validated data IO, features, model, evaluation, CLI
configs/          frozen protocol and raw-data checksums
tests/            leakage, threshold, export, grouping and data-contract tests
results/20261114/ full metrics, model exports and compressed decisions
report/           ACL-format paper, LaTeX source, and experiment notes
scripts/          download, independent verification and report builder
docs/             reproduction and provenance
```

Raw ARNOR/NYT text is excluded. Download it from the [authors' official release](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-ARNOR). A source-file checksum mismatch stops reproduction rather than silently accepting another dataset version.

## Limitations and next steps

- Small labeled development set, rare relations, strict span exclusions, and prior exposure to the corpus limit generalization claims.
- Adding handcrafted rules to the context model has not shown a clear independent gain.
- The noise-focused diagnostic exposes distribution sensitivity.
- Fresh annotations and a downstream extractor with matched-size random-filter controls are still needed to establish utility beyond this validity-filter task.
- The repository presents the consolidated project in one release commit; experimental comparisons and reproducibility records remain available.

Reference: Jia et al., [ARNOR: Attention Regularization based Noise Reduction for Distant Supervision Relation Classification](https://aclanthology.org/P19-1135/), ACL 2019.
