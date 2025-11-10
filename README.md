# PGC
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17564899.svg)](https://doi.org/10.5281/zenodo.17564899) ![GitHub License](https://img.shields.io/github/license/pinto0309/pgc) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/pgc)

Ultrafast pointing gesture classification. Classify whether the finger is pointing near the center of the camera lens.

A model that can only detect slow human gestures is completely worthless. A resolution of 32x32 is sufficient for human hand gesture classification. LSTM and 3DCNN are useless because they are not robust to environmental noise.

https://github.com/user-attachments/assets/19268cf9-767c-441e-abc0-c3abd8dba57a

|Variant|Size|F1|CPU<br>inference<br>latency|ONNX|
|:-:|:-:|:-:|:-:|:-:|
|S|494 KB|0.9524|0.43 ms|[Download](https://github.com/PINTO0309/PGC/releases/download/onnx/pgc_s_32x32.onnx)|
|C|875 KB|0.9626|0.50 ms|[Download](https://github.com/PINTO0309/PGC/releases/download/onnx/pgc_c_32x32.onnx)|
|M|1.7 MB|0.9714|0.59 ms|[Download](https://github.com/PINTO0309/PGC/releases/download/onnx/pgc_m_32x32.onnx)|
|L|6.4 MB|0.9782|0.78 ms|[Download](https://github.com/PINTO0309/PGC/releases/download/onnx/pgc_l_32x32.onnx)|

## Setup

```bash
git clone https://github.com/PINTO0309/PGC.git && cd PGC
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

## Inference

```bash
uv run python demo_pgc.py \
-pm pgc_l_32x32.onnx \
-v 0 \
-ep cuda \
-dlr

uv run python demo_pgc.py \
-pm pgc_l_32x32.onnx \
-v 0 \
-ep tensorrt \
-dlr
```

## Feature Introspection

### Heatmap Visualization (`10_visualize_pgc_heatmaps.py`)

```bash
uv run python 10_visualize_pgc_heatmaps.py \
--model pgc_l_32x32.onnx \
--image data/cropped/100000011/pointing1_00005922_1.png \
--output-dir feature_heatmaps_pointing
```

- The script clones the ONNX graph in memory, taps every Sigmoid→Mul activation, and runs a single forward pass for the specified image (32×32 RGB crops are expected; non-32 inputs are resized automatically).
- Each tapped tensor is reduced to a 2-D attention map (default channel mean), coloured with the `turbo` colormap, overlaid on the original crop, and written to `feature_heatmaps_pointing/00X_<layer>.png`.
- Saved PNGs include three panels (original/heatmap/overlay) and are post-resized to 30 % of the Matplotlib canvas, so the folder is easy to scan even with dozens of layers.
- Interpretation tips:
- Use `--invert-cmap` if you prefer bright colours for strong activations; by default the original (non-inverted) palette is used.
  - Brighter regions in the heatmap panel highlight where that block focuses after the Sigmoid gating.
  - Comparing overlays across layers shows how spatial focus sharpens as the network goes deeper; early layers may track the palm outline, while later ones tend to lock onto fingertips.
  - Use `--layers features.4` or `--limit 8` to focus on specific blocks, and re-run with other crops (e.g., negatives) to understand false positives.
  - Supply `--composite-topk 6` to auto-create a 3×2 collage (row- or column-major via `--composite-layout`) and choose between intensity-sorted or user-defined ordering with `--composite-sort`; the combined PNG includes a legend bar that maps colour to “Low/High impact.”

### Feature Ablation Statistics (`11_ablate_pgc_features.py`)

```bash
uv run python 11_ablate_pgc_features.py \
--model pgc_l_32x32.onnx \
--image-dir data/cropped/100000007 \
--max-images 1000
```
```
[IMAGE 0001] data/cropped/100000007/pointing1_00001826_1.png: baseline=1.0000
[IMAGE 0002] data/cropped/100000007/pointing1_00001827_1.png: baseline=1.0000
[IMAGE 0003] data/cropped/100000007/pointing1_00001828_1.png: baseline=1.0000
 :
 :

Processed 1000 images. Baseline prob: mean=0.9932 std=0.0717 min=0.0001 max=1.0000

Aggregate feature impact (sorted by mean Δ):
01. Mul      /base_model/features/features.2/block/block.2/Mul (/base_model/features/features.2/block/block.2/Mul_output_0) -> meanΔ=+0.9932 stdΔ=0.0717 meanProb=0.0000
02. Mul      /base_model/features/features.2/block/block.5/Mul (/base_model/features/features.2/block/block.5/Mul_output_0) -> meanΔ=+0.9932 stdΔ=0.0717 meanProb=0.0000
03. Mul      /base_model/features/features.2/block/block.6/Mul (/base_model/features/features.2/block/block.6/Mul_output_0) -> meanΔ=+0.9932 stdΔ=0.0717 meanProb=0.0000
04. Mul      /base_model/features/features.2/out_act/Mul (/base_model/features/features.2/out_act/Mul_output_0) -> meanΔ=+0.9932 stdΔ=0.0717 meanProb=0.0000
05. Mul      /base_model/stem/stem.5/Mul (/base_model/stem/stem.5/Mul_output_0) -> meanΔ=+0.9932 stdΔ=0.0717 meanProb=0.0000
06. Mul      /base_model/stem/stem.2/Mul (/base_model/stem/stem.2/Mul_output_0) -> meanΔ=+0.9932 stdΔ=0.0717 meanProb=0.0000
07. Mul      /base_model/features/features.3/out_act/Mul (/base_model/features/features.3/out_act/Mul_output_0) -> meanΔ=+0.9931 stdΔ=0.0717 meanProb=0.0000
08. Mul      /base_model/features/features.1/out_act/Mul (/base_model/features/features.1/out_act/Mul_output_0) -> meanΔ=+0.9921 stdΔ=0.0717 meanProb=0.0010
 :
 :
[INFO] Rendering heatmaps for top-6 tensors -> ablation_heatmaps/ablation_top_features.png
[OK] Saved /base_model/stem/stem.2/Mul -> ablation_heatmaps/001_base_model_stem_stem_2_Mul.png
[OK] Saved /base_model/stem/stem.5/Mul -> ablation_heatmaps/002_base_model_stem_stem_5_Mul.png
[OK] Saved /base_model/features/features.2/block/block.2/Mul -> ablation_heatmaps/003_base_model_features_features_2_block_block_2_Mul.png
[OK] Saved /base_model/features/features.2/block/block.5/Mul -> ablation_heatmaps/004_base_model_features_features_2_block_block_5_Mul.png
[OK] Saved /base_model/features/features.2/block/block.6/Mul -> ablation_heatmaps/005_base_model_features_features_2_block_block_6_Mul.png
[OK] Saved /base_model/features/features.2/out_act/Mul -> ablation_heatmaps/006_base_model_features_features_2_out_act_Mul.png
Model prob_pointing=1.0000
[COMPOSITE] Saved top-6 heatmaps -> ablation_heatmaps/ablation_top_features.png
```

- This utility clones the model, injects a learnable gate after each Sigmoid→Mul tensor (or any tensor selected via filters), and measures how much `prob_pointing` falls when each gate is zeroed.
- With `--image-dir` the script walks the directory (lexicographic order), processes up to `--max-images` crops, and reuses a single ONNX Runtime session, so running 100 images is practical.
- Increase `--max-images` (e.g., 1000) to cover more crops; the script simply takes the first N entries after sorting paths.
- Console output includes:
  - Per-image baselines when multiple inputs are provided (`[IMAGE 012] ... baseline=0.8531`), summarised at the end with mean/std/min/max.
  - An aggregate ranking (`meanΔ`, `stdΔ`, `meanProb`) showing how strongly each tensor contributes across the sampled set. Large positive `meanΔ` means the score collapses without that feature, indicating heavy reliance on that block.
  - In single-image mode, the script prints every `[ABLATE]` line so you can trace exactly which feature map drove the prediction up or down.
- Practical interpretation:
  - Compare rankings across pointing vs. non-pointing folders to see which stages distinguish positives; e.g., if `features.2/out_act` always yields `Δ≈1.0`, that stage is critical.
  - High `stdΔ` flags tensors whose influence varies sharply; inspect those cases with the heatmap script to spot context-specific attention.
  - Negative or near-zero deltas imply the network barely uses that tensor for the sampled images, hinting at redundancy or opportunities for pruning.
- By default the top 6 tensors (highest mean Δ) are auto-visualised via `10_visualize_pgc_heatmaps.py` using the first processed crop; their heatmaps plus a 3×2 collage (filled column-by-column so the most influential maps occupy the left column) land under `ablation_heatmaps/`. Override or disable this behaviour with `--topk-heatmaps 0`, `--topk-image`, or `--topk-output-dir`.

  |Label|Heatmaps|
  |:-:|:-:|
  |`not_pointing`|<img width="700" alt="ablation_top_features" src="https://github.com/user-attachments/assets/0b24c1dc-3ba1-44a8-817b-d894a7b0f361" />|
  |`pointing`|<img width="700" alt="ablation_top_features" src="https://github.com/user-attachments/assets/bdf1fff9-5cad-4b51-8fff-553e0b23da7b" />|

## Dataset Preparation

https://gibranbenitez.github.io/IPN_Hand/ CC BY 4.0

|id|Label|Gesture|Instances|
|-:|:-|:-|-:|
|1|D0X|Non-gesture|1,431|
|**2**|**B0A**|**Pointing with one finger**|**1,010**|
|3|B0B|Pointing with two fingers|1,007|
|4|G01|Click with one finger|200|
|5|G02|Click with two fingers|200|
|6|G03|Throw up|200|
|7|G04|Throw down|201|
|8|G05|Throw left|200|
|9|G06|Throw right|200|
|10|G07|Open twice|200|
|11|G08|Double click with one finger|200|
|12|G09|Double click with two fingers|200|
|13|G10|Zoom in|200|
|14|G11|Zoom out|200|

```bash
uv run python 01_prepare_pointing_dataset.py

pointing (class 1): 221193
not_pointing (class 0): 368756
Labels CSV: ./data/dataset/labels.txt
Pie chart: ./data/dataset/class_balance.png
```

```bash
uv run python 02_wholebody34_data_extractor.py \
-ha \
-ep tensorrt

Hand-only detection summary
  Total images: 600325
  Images with detection: 587443
  Images without detection: 12882
  Images with >=2 detections: 37657
  Crops per label:
    not_pointing: 333467
    pointing: 204065
```

<img width="800" alt="pointing_hand_size_hist" src="https://github.com/user-attachments/assets/f793764c-76cf-4d75-a3ca-0e0a51be3fa6" />

<img width="800" alt="not_pointing_hand_size_hist" src="https://github.com/user-attachments/assets/0fd01538-3065-489d-82b0-2cd684a9053f" />

```bash
uv run python 03_dataset_convert_to_parquet.py \
--annotation data/cropped/annotation.csv \
--output data/dataset.parquet \
--train-ratio 0.8 \
--seed 42 \
--embed-images

Split summary: {
  'train_total': 434377,
  'train_not_pointing': 271125,
  'train_pointing': 163252,
  'val_total': 108595,
  'val_not_pointing': 67782,
  'val_pointing': 40813,
}
Saved dataset to data/dataset.parquet (542972 rows).
```

Generated parquet schema (`split`, `label`, `class_id`, `image_path`, `source`):
- `split`: `train` or `val`, assigned with an 80/20 stratified split per label.
- `label`: string hand state (`pointing`, `not_pointing`); inferred from filename or class id.
- `class_id`: integer class id (`0` not_pointing, `1` pointing) maintained from the annotation.
- `image_path`: path to the cropped PNG stored under `data/cropped/...`.
- `source`: `train_dataset` for `000000001`-prefixed folders, `real_data` for `100000001`+, `unknown` otherwise.
- `image_bytes` *(optional)*: raw PNG bytes for each crop when `--embed-images` is supplied.

Rows are stratified within each label before concatenation, so both splits keep similar pointing/not_pointing proportions. Class counts per split are printed when the conversion script runs.

## Training Pipeline

- Use the images located under `dataset/output/002_xxxx_front_yyyyyy` together with their annotations in `dataset/output/002_xxxx_front.csv`.
- Every augmented image that originates from the same `still_image` stays in the same split to prevent leakage.
- The training loop relies on `BCEWithLogitsLoss` plus class-balanced `pos_weight` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities. Use `--train_resampling weighted` to switch on the previous `WeightedRandomSampler` behaviour, or `--train_resampling balanced` to physically duplicate minority classes before shuffling.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `pgc_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `pgc_best_epoch0004_f1_0.9321.pt` (also latest 10).
- The backbone can be switched with `--arch_variant`. Supported combinations with `--head_variant` are:

  | `--arch_variant` | Default (`--head_variant auto`) | Explicitly selectable heads | Remarks |
  |------------------|-----------------------------|---------------------------|------|
  | `baseline`       | `avg`                       | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, you need to adjust the height and width of the feature map so that they are divisible by `--token_mixer_grid` (if left as is, an exception will occur during ONNX conversion or inference). |
  | `inverted_se`    | `avgmax_mlp`                | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, it is necessary to adjust `--token_mixer_grid` as above. |
  | `convnext`       | `transformer`               | `avg`, `avgmax_mlp`, `transformer`, `mlp_mixer` | For both heads, the grid must be divisible by the feature map (default `3x2` fits with 30x48 input). |
- The classification head is selected with `--head_variant` (`avg`, `avgmax_mlp`, `transformer`, `mlp_mixer`, or `auto` which derives a sensible default from the backbone).
- Mixed precision can be enabled with `--use_amp` when CUDA is available.
- Resume training with `--resume path/to/pgc_epoch_XXXX.pt`; all optimiser/scheduler/AMP states and history are restored.
- Loss/accuracy/F1 metrics are logged to TensorBoard under `output_dir`, and `tqdm` progress bars expose per-epoch progress for train/val/test loops.

Baseline depthwise-separable CNN:

```bash
uv run python -m pgc train \
--data_root data/dataset.parquet \
--output_dir runs/pgc \
--epochs 100 \
--batch_size 256 \
--train_ratio 0.8 \
--val_ratio 0.2 \
--train_resampling balanced \
--image_size 32x32 \
--base_channels 32 \
--num_blocks 4 \
--arch_variant baseline \
--seed 42 \
--device auto \
--use_amp
```

Inverted residual + SE variant (recommended for higher capacity):

```bash
uv run python -m pgc train \
--data_root data/dataset.parquet \
--output_dir runs/pgc_is_s \
--epochs 100 \
--batch_size 256 \
--train_ratio 0.8 \
--val_ratio 0.2 \
--train_resampling balanced \
--image_size 32x32 \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--seed 42 \
--device auto \
--use_amp
```

ConvNeXt-style backbone with transformer head over pooled tokens:

```bash
uv run python -m pgc train \
--data_root data/dataset.parquet \
--output_dir runs/pgc_convnext \
--epochs 100 \
--batch_size 256 \
--train_ratio 0.8 \
--val_ratio 0.2 \
--train_resampling balanced \
--image_size 32x32 \
--base_channels 32 \
--num_blocks 4 \
--arch_variant convnext \
--head_variant transformer \
--token_mixer_grid 2x2 \
--seed 42 \
--device auto \
--use_amp
```

- Outputs include the latest 10 `pgc_epoch_*.pt`, the latest 10 `pgc_best_epochXXXX_f1_YYYY.pt` (highest validation F1, or training F1 when no validation split), `history.json`, `summary.json`, optional `test_predictions.csv`, and `train.log`.
- After every epoch a confusion matrix and ROC curve are saved under `runs/pgc/diagnostics/<split>/confusion_<split>_epochXXXX.png` and `roc_<split>_epochXXXX.png`.
- `--image_size` accepts either a single integer for square crops (e.g. `--image_size 48`) or `HEIGHTxWIDTH` to resize non-square frames (e.g. `--image_size 64x48`).
- Add `--resume <checkpoint>` to continue from an earlier epoch. Remember that `--epochs` indicates the desired total epoch count (e.g. resuming `--epochs 40` after training to epoch 30 will run 10 additional epochs).
- Launch TensorBoard with:
  ```bash
  tensorboard --logdir runs/pgc
  ```

### ONNX Export

```bash
uv run python -m pgc exportonnx \
--checkpoint runs/pgc_is_s/pgc_best_epoch0049_f1_0.9939.pt \
--output pgc_s.onnx \
--opset 17
```

- The saved graph exposes `images` as input and `prob_pointing` as output (batch dimension is dynamic); probabilities can be consumed directly.
- After exporting, the tool runs `onnxsim` for simplification and rewrites any remaining BatchNormalization nodes into affine `Mul`/`Add` primitives. If simplification fails, a warning is emitted and the unsimplified model is preserved.

## Arch

<img width="300" alt="pgc_s_32x32" src="https://github.com/user-attachments/assets/f6a6efcc-0b05-4cbe-b578-1c72312c1b61" />

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025pgc,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/PGC},
  month     = {11},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17564899},
  url       = {https://github.com/PINTO0309/pgc},
  abstract  = {Ultrafast pointing gesture classification.},
}
```

## Acknowledgements
- https://gibranbenitez.github.io/IPN_Hand/: CC BY 4.0 License
  ```bibtex
  @inproceedings{bega2020IPNhand,
    title={IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition},
    author={Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Yanai, Keiji},
    booktitle={25th International Conference on Pattern Recognition, {ICPR 2020}, Milan, Italy, Jan 10--15, 2021},
    pages={4340--4347},
    year={2021},
    organization={IEEE}
  }
  ```
- https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34: Apache 2.0 License
  ```bibtex
  @software{DEIMv2-Wholebody34,
    author={Katsuya Hyodo},
    title={Lightweight human detection models generated on high-quality human data sets. It can detect objects with high accuracy and speed in a total of 28 classes: body, adult, child, male, female, body_with_wheelchair, body_with_crutches, head, front, right-front, right-side, right-back, back, left-back, left-side, left-front, face, eye, nose, mouth, ear, collarbone, shoulder, solar_plexus, elbow, wrist, hand, hand_left, hand_right, abdomen, hip_joint, knee, ankle, foot.},
    url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34},
    year={2025},
    month={10},
    doi={10.5281/zenodo.10229410}
  }
  ```
- https://github.com/PINTO0309/bbalg: MIT License
