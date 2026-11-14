# Retaining Useful Supervision in Lightweight Relation Denoising

Zijun Wu | Experimental report | 14 November 2026

The project began in an NYU NLP course with Zijun Wu, Tianchen Xie, and Tianluo Zhang. The implementation and evaluation documented here were prepared with coding-agent assistance; contributions are recorded in docs/PROVENANCE.md.

## Abstract

A filter can achieve high noise recall by discarding most of its useful supervision. We revisit a hand-written linguistic filter and compare six lightweight models under an explicit clean-retention objective. A sparse L2 logistic model combines the original rules with entity-masked local words and bigrams, candidate-relation indicators, and relation-feature interactions. Five-fold nested sentence-grouped evaluation on 4,241 labeled development examples yields 89.93% clean retention and 31.84% noise recall, compared with 90.94% and 19.58% for a rule-feature logistic baseline. On 3,966 diagnostic test examples without development-sentence overlap, the new model retains 92.37% of clean examples and removes 30.44% of noise. A separate, smaller noise-focused diagnostic does not show the same improvement. The evidence supports a better measured retention-quality trade-off, not a neural benchmark win or a downstream relation-extraction claim.

## 1. Research question and rationale

The task is to decide whether a supplied relation label is supported by a sentence mentioning two entities. An incorrect NA label is also noise: NA does not automatically mean an example is useless. We therefore evaluate both NA and non-NA examples and select separate operating thresholds for these groups.

The previous recall-first objective encouraged removal of roughly 90% of labeled noise even when correct examples were also removed. This release separates two questions: does a model rank label validity better, and which operating point preserves useful data? Lowering the removal rate alone is not evidence of a better model.

## 2. Model progression

Hand-written rules score entity distance, intervening-text length, relation keywords, and entity order, and penalize uncertainty, list-like text, and overlapping entities. Rule-feature logistic regression learns how to combine eight such components and an NA indicator. The new sparse model additionally conditions on the candidate relation, learns relation-specific rule weights, and reads entity-masked local unigrams and bigrams. Its coefficients remain inspectable; no pretrained encoder or GPU is required.


---

## 3. Data and evaluation protocol

Data: the authors' ARNOR 2.0 corpus. This release is a supervised validity-filter experiment using labeled development data; it is not a reimplementation of ARNOR's distant-supervision training protocol. All files are identified by SHA-256 in configs/data_manifest.json. Raw news articles are not redistributed.

| Split | Raw mentions | Valid labeled, unique | Sentence groups | After dev-text exclusion |
| --- | --- | --- | --- | --- |
| Development | 4567 | 4241 | 1495 | - |
| Test | 4484 | 4050 | 1486 | 3966 |
| Noise-focused test | 533 | 301 | 157 | 297 |

Entity spans must resolve uniquely and match the text. We quarantine 326 development, 425 test, and 200 noise-focused mentions with ambiguous spans. We remove 9 duplicate test annotations and 26 duplicate noise-focused annotations; 4 unknown and 2 conflicting noise-focused labels are excluded. Counts are disclosed rather than silently treated as clean. Production inference emits an explicit abstention for uncertain spans.

Primary evaluation: five outer folds grouped by normalized sentence text; all entity mentions of a sentence stay together. Within each outer training set, three inner grouped folds select C from {0.1, 1, 10} by minimum log loss, then select thresholds from inner predictions. Vectorizer vocabulary and scalers fit only on each training partition. L2 logistic regression uses liblinear; the rule-only comparator uses scaled rule features, while sparse variants use maximum-absolute scaling. All models use the same folds, search budget, and decision objectives.

Retain90 maximizes noise recall subject to at least 90% clean retention separately for NA/non-NA on inner predictions. Retain95 is a conservative alternative. Recall90 maximizes clean retention subject to 90% noise recall. Thresholds use exact tied-score boundaries. These are empirical selection constraints, not guarantees that every outer fold or new population will meet the target.

The protocol was saved before the run. All final models, thresholds, and the development-based promotion decision were frozen before test labels were read in this run. The context model passed the predeclared development gate. However, official development and test data were examined in earlier project iterations. Neither this test nor the development OOF estimate is advertised as a pristine external evaluation. Model selection based on OOF evidence also creates selection uncertainty.

## Metric definitions

Noise is the positive class. TP = noisy example removed; FP = clean example removed; FN = noisy example retained; TN = clean example retained. Noise recall = TP/(TP+FN); clean retention = TN/(TN+FP); retained-set purity = TN/(TN+FN); noise precision = TP/(TP+FP). F1 refers only to noise detection. Undefined denominators remain null. These metrics cannot be substituted for relation-classification F1.


---

## 4. Primary development results

All values below except AUC are percentages. These are pooled outer-fold predictions, not fitted training performance. Keep-all would retain 100% of clean examples, remove 0% of noise, and have 75.19% purity on this development set.

| Model | Clean retained | Noise removed | Kept purity | ROC AUC |
| --- | --- | --- | --- | --- |
| Hand-weighted rules | 97.99 | 2.00 | 75.19 | 0.550 |
| Rule-feature LR | 90.94 | 19.58 | 77.42 | 0.643 |
| Relation prior only | 97.02 | 12.55 | 77.08 | 0.562 |
| Rules + relation LR | 90.22 | 27.85 | 79.13 | 0.681 |
| Context + relation LR | 89.97 | 31.75 | 79.98 | 0.743 |
| Rules + context + relation LR | 89.93 | 31.84 | 80.00 | 0.745 |

At approximately 90% clean retention, the new context model removes 12.26 percentage points more noise than rule-feature LR. A paired bootstrap over sentence groups gives a conditional 95% interval of +8.72 to +15.94 points. Clean retention differs by -1.00 points (interval -3.01 to +1.14). The interval conditions on fitted OOF predictions; it does not include retraining or design-selection variability.

The raw hand-weighted score keeps 97.99% of clean examples and removes only 2.00% of noise. Score ties, especially for NA, prevent a deterministic threshold from spending the full allowed rejection budget. This row is not matched to the others at exactly the same retention; its threshold-independent AUC is also reported.

## 5. Main diagnostic test

This table excludes the 84 eligible mentions whose normalized sentence also occurs in development. It contains 2,987 clean and 979 noisy examples (3,966 total). Thresholds and model choices are fixed from development.

| Model | Clean retained | Noise removed | Kept purity | ROC AUC |
| --- | --- | --- | --- | --- |
| Hand-weighted rules | 97.09 | 2.04 | 75.15 | 0.553 |
| Rule-feature LR | 90.36 | 21.86 | 77.92 | 0.661 |
| Relation prior only | 99.90 | 2.25 | 75.72 | 0.559 |
| Rules + relation LR | 90.93 | 24.72 | 78.66 | 0.708 |
| Context + relation LR | 91.80 | 32.18 | 80.50 | 0.758 |
| Rules + context + relation LR | 92.37 | 30.44 | 80.20 | 0.761 |

The selected model retains 2,759 clean examples and removes 298 noisy examples; rule-feature LR retains 2,699 and removes 214. Thus, on this diagnostic set the new model preserves 60 additional clean examples while removing 84 additional noisy examples. This is an improvement in both measured outcomes under the frozen policies, rather than a direct numerical comparison with the historical 14.30% retention run, which used a different objective and sample handling.


---

## 6. What the ablations establish

Learning rule weights improves AUC over a hand-weighted total. Allowing relation-specific coefficients further raises development AUC from 0.643 to 0.681. Adding masked lexical context raises it to 0.745. Compared with rules plus relation features, the context model increases OOF noise recall by 3.99 points with a sentence-bootstrap interval of +0.91 to +7.26, at similar clean retention. Candidate-relation priors alone do not explain the full improvement.

Removing the hand-designed rule features from the context model produces nearly identical OOF performance: 31.75% versus 31.84% noise recall. The paired difference interval includes zero. The evidence therefore does not establish that the original rules add a substantial gain once local context is available. The rules remain valuable as a transparent starting hypothesis and an interpretable baseline, rather than being credited with an unsupported causal contribution.

## 7. Operating points and the retention problem

| Method / objective | Dev clean retained | Dev noise removed | Test clean retained | Test noise removed |
| --- | --- | --- | --- | --- |
| Rule-feature LR / recall90 | 18.53 | 91.35 | 26.55 | 88.97 |
| Rules + context + relation LR / recall90 | 42.14 | 88.88 | 38.13 | 91.52 |
| Rule-feature LR / retain90 | 90.94 | 19.58 | 90.36 | 21.86 |
| Rules + context + relation LR / retain90 | 89.93 | 31.84 | 92.37 | 30.44 |
| Rules + context + relation LR / retain95 | 94.76 | 21.48 | 96.02 | 16.34 |

Changing the objective explains part of the large increase in retained clean data. Improved discrimination explains another part: under the same Recall90 selection rule, the context model retains 38.13% of clean examples on the main diagnostic versus 26.55% for rule-feature LR, while removing 91.52% versus 88.97% of noise. Under Retain95, the new model preserves 96.02% of clean data but removes only 16.34% of noise. No single threshold simultaneously solves both objectives.

Plot data: policy_tradeoffs.csv. Lines connect three development-selected operating points; they are not a complete frontier.

Three development-selected operating points on the main diagnostic; lines are visual guides, not interpolated deployment policies.


---

## 8. Robustness and failure analysis

On the 2,735 main test examples with neither a development sentence nor a development entity pair, the context model retains 94.04% of clean examples and removes 32.87% of noise. Rule-feature LR retains 90.40% and removes 18.22%. This sensitivity analysis reduces one source of overlap; it does not establish generalization to another corpus or resolve prior human exposure to the test set.

| Main test subgroup | Model | Clean retained | Noise removed |
| --- | --- | --- | --- |
| NA | Rule-feature LR | 90.70 | 24.73 |
| NA | Rules + context + relation LR | 93.19 | 28.36 |
| non-NA | Rule-feature LR | 89.31 | 12.77 |
| non-NA | Rules + context + relation LR | 89.85 | 37.02 |

The small noise-focused diagnostic has a different composition: 183 of its 297 eligible non-overlapping examples are labeled noise, compared with 979 of 3,966 in the main diagnostic. It also loses many examples under strict span validation. We report it as a separate stress case, not as a directly comparable ARNOR benchmark reproduction.

| Noise-focused method | Clean retained | Noise removed | Noise F1 | ROC AUC |
| --- | --- | --- | --- | --- |
| Hand-weighted rules | 91.23 | 39.34 | 54.34 | 0.494 |
| Rule-feature LR | 88.60 | 41.53 | 55.88 | 0.788 |
| Context + relation LR | 90.35 | 36.07 | 50.77 | 0.654 |
| Rules + context + relation LR | 91.23 | 33.88 | 48.63 | 0.664 |

The context model is weaker than rule-feature LR on this stress case: 33.88% versus 41.53% noise recall. Its AUC is also lower (0.664 versus 0.788). A plausible explanation is that the lexical features fit the development distribution better than this selected noise set, but this experiment does not isolate the cause. No architecture or threshold was retuned after seeing these outcomes.

## 9. CPU footprint and engineering

| Model | Examples | Median seconds | JSON bytes |
| --- | --- | --- | --- |
| Rule-feature LR | 4050 | 0.154 | 1,042 |
| Rules + context + relation LR | 4050 | 0.255 | 4,093,035 |

Timing includes feature extraction, portable JSON scoring, and decisions on 4,050 validated examples. It excludes file IO. There was one warm-up and four measured repeats, with one numerical-library thread requested. These local CPU measurements do not support a speedup ratio against BERT or ARNOR. Model coefficients are plain JSON rather than executable pickle. Invalid spans generate abstentions in the inference CLI.


---

## 10. Interpretation and next experiments

The experiments support a bounded conclusion: relation-conditioned local context improves this lightweight validity filter on the development evaluation and the main diagnostic. The progression from hand-written linguistic hypotheses to learned coefficients and local lexical evidence separates representation quality from operating-point selection. The context-only ablation suggests that lexical features explain most of the additional gain; the noise-focused diagnostic shows that the improvement is distribution-dependent. This release establishes a reproducible supervised sparse baseline, not state-of-the-art performance.

Limitations: the labeled development set is small, many rare relations have very few examples, ambiguous spans are excluded, outer-fold retention fluctuates, and the corpus has been inspected before. Confidence intervals are conditional, the probabilities are not externally calibrated, and choosing a model from development OOF results adds selection uncertainty. The context-only ablation weakens any claim that handcrafted rules are essential. The specialized noise set reveals distribution sensitivity.

The next validation should be fixed before obtaining new results: independently annotate a fresh sentence-grouped holdout; examine rare-relation and domain shift; compare downstream relation extractors trained on all data, filtered data, and matched-size random subsets; and evaluate abstention/review and probability calibration. No downstream gains or neural comparisons are claimed in this release.

## 11. Reproduction and provenance

Install the package with the locked numerical dependencies. Download the official corpus using scripts/download_data.py, which verifies all four file hashes. Run python -m relden.experiment with a new results directory. Run scripts/verify_release.py to independently recalculate the saved metrics, verify sentence-group isolation, and check the frozen model hashes. The tests cover label leakage, entity masking, tied-score thresholds, export parity, grouped folds, and handling of ambiguous/unknown/conflicting records.

The project progressed from hand-designed rules to learned weights and context-sensitive filtering. Team contributions and coding-agent assistance are documented in docs/PROVENANCE.md. The final repository consolidates the code and results into one release commit, with earlier history backed up separately. Only results produced by the current evaluation scripts are used in this report.

## References

[1] Jia, W., Dai, D., Xiao, X., and Wu, H. (2019). ARNOR: Attention Regularization based Noise Reduction for Distant Supervision Relation Classification. ACL, 1399-1408. https://aclanthology.org/P19-1135/

[2] ARNOR authors' official dataset and version-2 documentation. https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2019-ARNOR

[3] Scikit-learn documentation: grouped cross-validation and decision-threshold tuning. https://scikit-learn.org/stable/modules/cross_validation.html ; https://scikit-learn.org/stable/modules/classification_threshold.html

[4] Project source history: https://github.com/LewisZZZ-Zhang/NLP_Final_Project ; consolidated release: https://github.com/JonathanWu04/interpretable-relation-denoising
