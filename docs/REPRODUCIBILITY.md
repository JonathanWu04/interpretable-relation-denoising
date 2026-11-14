# Reproducing the release

Use Python 3.12 for the locked reproduction environment. Runtime measurements were taken on the local machine recorded in `results/20261114/environment.json`; they are not a cross-hardware speed comparison.

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

The download checks the SHA-256 hash of each source file. The runner refuses to overwrite an existing results directory. It writes the protocol before fitting and saves all final configurations and the development-based selection before opening test labels. Feature vocabulary, scaling, C selection, and threshold selection all stay on the training side of each outer fold. The run fixes a seed and requests one numerical-library thread.

The release uses six methods, three C values, five outer folds, three inner folds, and three operating policies. All learned methods choose C by inner log loss. Thresholds are chosen independently for NA/non-NA. Retain90 and retain95 target clean retention on inner predictions; they are not statistical guarantees for unseen data. Recall90 exposes the aggressive trade-off used in earlier experiments.

`results/20261114/predictions/*.json.gz` contain deterministic gzip copies of the full decision records. `verify_release.py` works on either compressed release files or uncompressed fresh outputs, recomputes each metric table, checks group isolation, and verifies model hashes. Timings may vary. The inferential confidence intervals condition on the saved OOF predictions rather than refitting the model for every bootstrap sample.

## Inference

```sh
relden --model results/20261114/models/context_lr.json \
  --input data/arnor/test.json --output decisions.jsonl --policy retain90
```

Input follows the authors' ARNOR sentence JSONL format with `sentText` and `relationMentions`. Gold `is_noise` values are unnecessary for inference and do not enter features. The output contains one decision per original relation mention. Missing or ambiguous entity spans cause an explicit `abstain`, rather than being silently labeled clean or dropped. Output paths must be new.

The model stores coefficients and thresholds in JSON; inference never unpickles executable objects. Feature names are human-readable. The rule-only model is also available for inspection and comparison. The default model is selected from development evidence, not from the diagnostic test scores.

## Rebuilding the report

```sh
python -m pip install '.[report]'
python scripts/build_report.py --engine tectonic
```

This regenerates the Markdown report and PDF from saved metrics, and writes plot-ready CSV data. Report sources, references, and technical limitations are included. For an exact reproduction, pass `--results results/reproduction` and `--out report/reproduction`. The prose was reviewed for this release: a metric-signature check rejects changed results rather than silently pairing new tables with old interpretations. Review the narrative before updating that signature for a new experiment.

## Optional GitHub Actions

`docs/ci_workflow.example.yml` is an inactive example for the same tests and evidence verification. The upload credential lacks the GitHub workflow scope, so this release does not install or claim a passing hosted workflow. Local tests and a fresh-environment complete reproduction were performed.

The manuscript uses the ACL style supplied with the original project paper. Install Tectonic or upload all files in `report/latex/` to Overleaf and compile `main.tex`. `scripts/build_report.py` compiles this editable source without retraining. To regenerate the detailed metric notes separately, use `python scripts/build_experiment_notes.py --out /tmp/relden-notes`; this preserves the final paper PDF.
