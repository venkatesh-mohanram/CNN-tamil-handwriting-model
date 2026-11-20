# Model, dataset and training details

## What is being run

This package uses a specialized **13-class image classifier**, based on torchvision **MobileNetV3-Small**, rather than Qwen or another generative vision-language model. It takes one symbol image and returns 13 class logits plus a normalized visual embedding. A separate validation-selected rejection rule can turn a prediction into “not recognized.” There is no learned fourteenth unknown class.

Ordered Unicode labels are **அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ ஃ**. The ONNX contains the complete fine-tuned network. It is not a LoRA adapter, does not require a separate base model, and uses no quantization.

| Property | Exact artifact |
|---|---|
| Release/model version | `tamil-symbols-0.1.0` |
| Architecture | MobileNetV3-Small with 13-class final linear layer |
| Initialization | `MobileNet_V3_Small_Weights.IMAGENET1K_V1`, pretrained on ImageNet-1K |
| Upstream pretrained filename | `mobilenet_v3_small-047dcff4.pth` |
| Total learned parameters | **1,531,181** |
| Feature-backbone parameters | 927,008 |
| Classifier parameters | 604,173 |
| Feature extractor | torchvision Small backbone: 11 inverted-residual blocks between initial/final convolution stages; 13 `features` modules total |
| Classifier | Linear 576→1024, Hardswish, **Dropout p=0.2**, Linear 1024→13 |
| Pooling / embedding | Global average pooling produces 576 features; L2-normalized embedding returned alongside logits |
| Computation | One backbone pass for both outputs; Float32 weights and inputs |
| ONNX | Opset **17**, 6,135,453 bytes, dynamic batch dimension |
| Input | `image`: Float32 `[N,3,224,224]`, channel-first RGB |
| Outputs | `logits`: `[N,13]`; `embedding`: `[N,576]` |
| Packaged inference | ONNX Runtime **1.24.4**, CPU execution provider |

The ONNX graph annotates the embedding's second dimension symbolically, but runtime/model validation requires 576 features. The original ImageNet checkpoint and ImageNet images are not required or included for inference. Model release files and their SHA-256 checksums are under `server/model/`; exact training records are copied below.

## Preprocessing version 2

The server is authoritative. It accepts one PNG, at most 1 MiB and 1024 pixels in either dimension; animated/invalid/blank/excessive-ink inputs are rejected. Transparent pixels are composited onto white. The image is converted to grayscale, its 75th-percentile intensity estimates the paper background, and intensities are scaled toward white. Pixels at or above 200 are whitened. This handles gray scanned paper without inverting light-on-dark drawings.

Foreground is every pixel below 200. The bounding box encloses **all components**, preserving ஃ dots and disconnected ஔ strokes. The entire crop is centered in a white square of side `max(crop width, crop height)+40` (20 original-pixel padding on each side), then resized to 224×224 with Lanczos interpolation. Grayscale is repeated into three RGB channels. Values are scaled to 0–1, converted to NCHW Float32 and normalized with ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.

No largest-component filtering, mirroring or stroke-order analysis is used. The drawing component exports the complete white-background image; it does not implement a different normalization pipeline.

## Source corpus and partitions

Dataset: **tamil-handwritten-character-carpus**, Mendeley Data, version 1 (2023), [DOI 10.17632/6zcpgchvmx.1](https://doi.org/10.17632/6zcpgchvmx.1), CC BY 4.0. Full creator list and processing attribution are in [ATTRIBUTION.md](ATTRIBUTION.md). Publisher dataset-specific terms were checked; a related paper's license was not substituted for dataset permission.

| Item | Count/details |
|---|---|
| Original archive | `tamil-alphabets-raw.rar`, 1,436,820,585 bytes |
| Archive SHA-256 | `e016b9a10c83922f60506b0c2cea218b23be98d87d8a8462a5601d6587d5a150` |
| Full archive | 24,700 PNGs, 247 folders, 100 per folder |
| Supported selection | Vowels in folders 1–12; ஃ in folder 247 |
| Negative selection | Unsupported human handwritten mei in folders 13–30 (18 classes) |
| Selected original images validated | 3,100 |
| Exact duplicates removed | 31 |
| Retained records | **3,069**: 1,287 supported positives and 1,782 negatives |
| Each supported symbol | 99 samples: **69 training / 15 validation / 15 test** |
| Supported totals | 897 training / 195 validation / 195 test |
| Negative totals | 1,242 training-partition / 270 validation / 270 test |
| Prepared manifest SHA-256 | `728120e7898c1f36e1b50c8591cd85566d458b46a051f2efc51c0f86ebafd49e` |

Original selected PNG bytes were preserved and checksummed in the training project. The inference handoff contains provenance, not raw corpus images. Grouped splitting is deterministic (seed 42), approximately 70/15/15 by available source-page proxies. Exact duplicates are removed before splitting; related page groups are joined, and dHash distance≤3 near-duplicate candidates are investigated/grouped conservatively. The actual selected set had zero such near-duplicate candidates; this does not prove no visual similarity exists.

Individual writer IDs and ages are unavailable; the publisher describes B.Tech student samples. These are **page-isolated**, not proven writer-isolated partitions. Children's handwriting, unseen-writer accuracy and canvas-domain performance remain unverified. Folder mapping was audited with source contact sheets and complete selected-image decoding; not every sample was individually visually adjudicated.

The 13-class classifier and class prototypes use **positive training examples only**. The negative training partition was retained for provenance but not used as a fourteenth class or in the base classification loss. Validation negatives selected rejection thresholds; test negatives measured rejection. No application submissions, volunteers, font-generated positive samples or newly collected handwriting were used. HP Labs data was excluded due to restrictive terms; uTHCD was not used because dataset-specific access/terms/mapping were unverified.

Source inventory and exact class/split counts: [source-inventory.json](server/docs/source-inventory.json), [source-coverage.json](server/docs/source-coverage.json).

## Fine-tuning method

This is **supervised transfer learning with full backbone fine-tuning**, not LoRA/QLoRA or prompt tuning. The final 1,000-way ImageNet output layer was replaced by a randomly initialized 13-way linear layer. The remaining ImageNet-pretrained network was retained as initialization.

1. **Head phase, epochs 1–2:** freeze feature-backbone parameters and its BatchNorm running statistics; train the classifier (604,173 parameters).
2. **Fine phase, epochs 3–22:** train the entire network (1,531,181 parameters). For the selected run, all BatchNorm modules remain in evaluation mode so their running mean/variance stay fixed; their learned affine parameters can still receive gradients. Dropout remains active at p=0.2 during training and is disabled for evaluation/inference.

| Setting | Value |
|---|---|
| Loss | Cross-entropy over 13 supported classes |
| Optimizer | AdamW |
| Head initial learning rate | 0.001 |
| Backbone-phase initial learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Schedule | CosineAnnealingLR, independently initialized for each phase; T_max=2 then20; stepped each epoch |
| Batch size | 32 |
| Seed | 42 for Python, NumPy and PyTorch |
| Device | Apple M1 Pro Mac, 32 GiB RAM, PyTorch MPS |
| Training software | Python3.12.13; torch2.10.0; torchvision0.25.0; NumPy2.2.6; Pillow12.3.0; safetensors0.7.0 |
| Selection metric | Highest validation macro recall; test data excluded from checkpoint/model selection |
| Selected run | `candidate-002-bn-frozen`, `freeze_batch_norm: true` |
| Selected checkpoint | **Epoch6**, validation macro recall **97.9487%** |

Augmentation applies only after splitting and only to training positives. Paper is whitened before padding. Each sampled augmentation uses rotation −10°…+10°, translation up to ±8% per axis, scale0.9…1.1, no shear/mirroring, bilinear sampling with white fill. Stroke variation uses a3×3 minimum filter with15% probability, maximum filter with15%, otherwise neither. It does not create new independent handwriting samples.

Both experimental runs completed22 epochs:

| Candidate | BatchNorm statistics during full fine-tuning | Best validation macro recall | Best epoch | Sum of epoch durations |
|---|---|---:|---:|---:|
| candidate-001 | Updated | 92.8205% |21|412.7495s|
| candidate-002-bn-frozen | Frozen | 97.9487% |6|402.0700s|

These runtimes exclude setup, persistence/copy overhead and export; they are not complete wall-clock training duration. The second candidate was selected from validation results before held-out evaluation. No further training or threshold tuning followed the reported test result.

Exact configuration, dependencies and per-epoch loss/validation history: [candidate001](server/docs/training-candidate-001.json), [selected candidate002](server/docs/training-candidate-002-bn-frozen.json), [selected checkpoint checksum](server/docs/selected-checkpoint.json).

The training project saves full model/optimizer/scheduler/epoch-step/RNG state after each epoch using temporary paths, a completion marker and checksums; incomplete checkpoints are ignored. It retains two complete recovery checkpoints and the best validation weights, with an optional verified Google Drive mirror. Local interruption/mirror-only recovery was tested. Real Colab eviction/Drive execution was not tested. Training source, recovery checkpoints, raw data and PyTorch are intentionally outside this inference handoff.

## Rejection method

For each class, L2-normalized training embeddings are summed and renormalized to a 576-dimensional class prototype. At inference, the highest-logit class is considered; a decision is accepted only if both its softmax probability and cosine similarity to that class's prototype exceed validation-selected thresholds.

Selected thresholds: **softmax≥0.3**, **cosine similarity≥0.6933333333333332**. A grid search on positive/negative validation data prioritizes 95% macro recall and90% minimum class recall where attainable, then negative rejection; otherwise it preserves the best attainable recall. These settings were frozen before test evaluation. Neither softmax nor the threshold rule is a calibrated confidence estimate, and the API does not expose them as one.

This heuristic is weak for unsupported Tamil mei: most are accepted. It should not be interpreted as reliable open-set recognition.

## Export and measured results

The selected model was loaded on CPU in evaluation mode and exported to ONNX opset17, with dynamic batch axes. The exported ONNX itself was evaluated. Maximum source/ONNX absolute logit difference over10 fixed held-out examples: **0.0000557899**.

| Test outcome | Measured result |
|---|---|
| Supported overall | **186/195 =95.3846%** |
| எ |12/15=80.00% — fails90% per-symbol gate|
| ஒ |10/15=66.67% — fails90% per-symbol gate|
| ஓ |14/15=93.33%|
| Other10 supported symbols |15/15=100% each|
| Valid supported-input rejection |0/195=0%|
| Unsupported human mei accepted |**260/270=96.2963%**|
| Supplementary synthetic scribbles accepted |17/100=17%, seed61013; not human handwriting and not used for threshold selection|

[Evaluation report](server/docs/model-evaluation.md) includes the confusion matrix. [Evaluation JSON](server/docs/model-evaluation.json) preserves counts, predictions, source grouping and calibration/selection context. No successful overall average can override failed per-symbol criteria.

## Qualification boundary

The candidate is **unqualified**; `qualified: false` remains unchanged. The original production service refuses to activate it. This handoff intentionally permits localhost manual verification with a visible experimental warning; that local mode is not production approval and does not change model metadata.

Ubuntu20-user/1,000-request p95≤2s,p99≤3s qualification has not been run. Docker execution, actual Colab/Drive recovery and complete touch/stylus/accessibility qualification remain unverified. Successful manual drawing predictions do not establish those gates, child accuracy or generalization to arbitrary canvas styles. Future improvement requires independent evaluation data rather than repeatedly tuning against this already-reported test set.
