# Candidate evaluation — not beta qualified

This report records the original production-gated evaluation. The portable handoff separately permits explicitly experimental localhost manual verification while preserving qualified=false; it does not change these metrics or approve production deployment.

Measured locally on 14 November 2026. Model: MobileNetV3-Small, 13 classes, ONNX Runtime CPU. Two training candidates used validation selection; the selected export was evaluated once on the held-out partition. No subsequent training or rejection tuning uses these test results.

## Outcome

**Overall: 186/195 (95.4%). The candidate fails the 90% per-symbol gate.** All valid rejections count as incorrect; there were zero valid rejections. The service refuses to activate this candidate.

| Symbol | Held-out samples | Correct | Accuracy |
|---|---:|---:|---:|
| அ | 15 | 15 | 100.0% |
| ஆ | 15 | 15 | 100.0% |
| இ | 15 | 15 | 100.0% |
| ஈ | 15 | 15 | 100.0% |
| உ | 15 | 15 | 100.0% |
| ஊ | 15 | 15 | 100.0% |
| எ | 15 | 12 | 80.0% |
| ஏ | 15 | 15 | 100.0% |
| ஐ | 15 | 15 | 100.0% |
| ஒ | 15 | 10 | 66.7% |
| ஓ | 15 | 14 | 93.3% |
| ஔ | 15 | 15 | 100.0% |
| ஃ | 15 | 15 | 100.0% |

Unsupported human handwritten Tamil mei: **260/270 accepted (96.3%)**. This rejection mechanism is inadequate for unsupported Tamil inputs. Separately, **17/100 synthetic scribbles were accepted (17%)**. These generated scribbles are supplementary test inputs, not human handwriting or child/canvas evidence. Softmax scores are not calibrated confidence.

## Training and provenance

Both runs used this Mac (Apple M1 Pro, 32 GiB, 8 CPUs), PyTorch MPS, seed42, 2 head epochs, 20 backbone epochs, batch32, pretrained weights, augmentation, AdamW, head LR0.001 and backbone LR0.0001. Candidate1 updated BatchNorm and reached92.8% validation macro recall. Candidate2 froze BatchNorm statistics and reached97.9%, selecting epoch6. Summed epoch runtimes were412.7s and402.1s respectively; these exclude setup, checkpoint-copy overhead and export.

All195 supported test examples and270 human negatives come from Mendeley version1. Each supported class has69 training,15 validation and15 test examples. Splits separate source-page proxies; writer IDs are unavailable. Children-specific, unseen-writer and canvas-specific performance are unverified. See [attribution](../../ATTRIBUTION.md) and [source audit](source-inventory.json).

ONNX/source maximum absolute logit difference on10 fixed held-out inputs: **0.00005579**. Full predictions, per-source summary, source-page references, configuration and confusion counts are in [model-evaluation.json](model-evaluation.json). Immutable artifacts and checksums are in [the candidate directory](../model/source-release.json).

## Confusion matrix

Rows are references; columns are predictions. No valid test input was rejected.

| Reference | அ | ஆ | இ | ஈ | உ | ஊ | எ | ஏ | ஐ | ஒ | ஓ | ஔ | ஃ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| அ | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ஆ | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| இ | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ஈ | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| உ | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ஊ | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| எ | 1 | 0 | 0 | 2 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 |
| ஏ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 | 0 |
| ஐ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 |
| ஒ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 5 | 0 | 0 |
| ஓ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 14 | 0 | 0 |
| ஔ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 |
| ஃ | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 15 |

## Remaining gates

The corpus recognition gate failed forஎ andஒ. The rejection method also needs improvement. Future model/rejection experiments require a newly reserved independent test partition or another verified corpus; do not repeatedly optimize against this reported test set. No volunteer collection or application drawing retention is proposed.

Ubuntu latency/load qualification, Docker execution, real Colab/Drive eviction recovery and full touch/stylus/accessibility qualification remain unverified. The localhost browser test verified mouse dots, Undo, Clear and safe unavailable-model handling through the real backend adapter. At original production-service validation, a successful-recognition UI flow and valid-model API benchmark were blocked by model qualification. The portable handoff provides a separate experimental manual-verification path; the production gate and Ubuntu benchmark remain unsatisfied.
