# CPI Bat Colony Counter — CSRNet

Automated bat colony counting system for wildlife monitoring and pandemic early warning, developed as part of the **NSF Center for Pandemic Insights (CPI)**. The system uses a convolutional neural network to estimate bat populations from thermal infrared emergence video without individual detection or tracking.

---

## Overview

Traditional bat counting methods — individual detection, multi-camera triangulation, manual annotation — break down in dense swarms where bats overlap and occlude each other at thousands per second. This project takes a density estimation approach: instead of tracking individuals, CSRNet learns to predict a *spatial density map* over each frame. The sum of that map is the bat count. An optical flow layer then measures directional bat flux across a virtual counting line to accumulate a total emergence population estimate across the full video.

---

## Architecture

### Model: CSRNet with VGG16 Frontend

```
Input Frame
    │
    ▼
Frontend — VGG16 conv layers 1–10 (pretrained ImageNet weights)
    │        [64, 64, MaxPool, 128, 128, MaxPool, 256, 256, 256, MaxPool, 512, 512, 512]
    ▼
Backend — 6 dilated conv layers (dilation rate=2, in_channels=512)
    │        [512, 512, 512, 256, 128, 64]
    ▼
Output — 1×1 conv → density map (H/8 × W/8)
```

The VGG16 frontend is fine-tuned at a lower learning rate (`1e-5`) while the backend and output layer train from scratch at `1e-4`. This transfer learning strategy lets the network reuse low-level texture features from ImageNet while learning bat-specific density patterns in the backend.

### Loss Function

Root MSE (Euclidean loss) between predicted and ground-truth density maps:

```python
loss = sqrt(MSE(pred, gt))
```

---

## Dataset

| Property | Value |
|---|---|
| Total annotated tiles | 10,960 |
| Training format | 160×128 meshed images |
| Density map kernel | Tight Gaussian, σ=0.5 |
| Density map resolution | H/8 × W/8 of input |
| Source footage | Thermal infrared bat emergence video |

**Meshed training images** tile multiple annotated patches into larger composite images (160×128), which improves batch diversity and training efficiency compared to using raw small crops.

Ground-truth density maps are stored as `.npy` files alongside images, either at full resolution (H×W) or pre-downsampled to H/8×W/8. The dataset loader handles both formats automatically.

### Data Augmentation (training only)

- Random crop (128×128)
- Color jitter (brightness ±0.3, contrast ±0.3, saturation ±0.2)
- Random Gaussian blur (kernel=3, σ∈[0.1, 1.0])
- Random horizontal flip
- Random vertical flip

---

## Training

### Hyperparameters

| Parameter | Value |
|---|---|
| Crop size | 128×128 |
| Batch size | 8 |
| Backend LR | 1e-4 |
| Frontend LR | 1e-5 |
| Weight decay | 1e-4 |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau (factor=0.7, patience=3) |
| Max epochs | 100 |

### Results

| Metric | Value |
|---|---|
| MAE (meshed test set) | **3.587** |

The best checkpoint is saved automatically whenever a new lowest MAE is achieved. Training history (loss, MAE, RMSE per epoch) is logged to `training_history.csv`.

### Running Training

Open `CPI_CSRNet.ipynb` in Google Colab with a GPU runtime and execute the cells under **Transfer Learning** sequentially. Mount your Google Drive and ensure the following folder structure exists:

```
CPI_Folder_CSRNet/
├── meshed/
│   ├── train/
│   │   ├── images/       # .png tiles
│   │   └── density/      # .npy density maps
│   └── test/
│       ├── images/
│       └── density/
├── video/
│   └── output.mp4        # source thermal video
└── logs/
    └── missing_maps      # auto-generated skip log (JSONL)
```

---

## Inference Pipeline

### Single Image

```python
# Load checkpoint
model = CSRNet(load_imagenet_frontend=False)
ckpt  = torch.load("csrnet_meshed_best.pth", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# Predict
pred, count = predict(image_path)           # returns count scalar
predict_and_plot(image_path)               # count + density heatmap side-by-side
```

### Video — Sliding Window Inference

Full frames are tiled into 128×128 patches (stride=128, no overlap). CSRNet runs on each patch independently; density maps are stitched back into a full-frame density map and summed for the total count.

```python
count, density_full = predict_frame_sw(frame_bgr, patch_size=128, stride=128)
```

Output per video run:
- **Annotated video** — original frames overlaid with a HOT colormap density heatmap and live bat count / timestamp text
- **CSV** — `frame_idx`, `timestamp_s`, `bat_count` per processed frame
- **Plot** — bat count over time with mean and max reference lines

### Region of Interest (ROI) Mode

For deployments where only part of the frame contains the emergence site, inference can be restricted to a manually specified bounding box. A grid-overlay helper (`get_frame_with_grid`) assists in identifying pixel coordinates:

```python
# Preview the ROI boundary
preview_roi(video_path, x1=1000, y1=100, x2=1280, y2=400)

# Run inference within ROI only
frame_counts = predict_video_with_roi(video_path, model, img_tf, device,
                                      x1=1000, y1=100, x2=1280, y2=400)
```

---

## Flux-Based Population Estimation

The core population estimate uses **Farneback optical flow** combined with per-frame density maps to measure net bat flux across a virtual vertical counting line at the horizontal midpoint of the ROI.

### How It Works

1. For each frame, compute a density map over the ROI (two 160×128 patches tiled in a 2×3 grid)
2. Compute Farneback optical flow between consecutive grayscale frames
3. Downsample the horizontal flow field (`u` component) to density map resolution
4. At the midpoint column: `flux = Σ (density_col × flow_col)` — bats with rightward flow add to the count; leftward flow subtracts
5. Accumulate flux across all frames → **total population estimate**

```python
df = compute_flux_roi(
    video_path="Y2.2.3[Val2].mov",
    model=model, img_tf=img_tf, device=device,
    roi_x=480, roi_y=34,
    roi_w=160, roi_h=256,          # must be divisible by 160 and 128
    output_csv="flux_roi_results.csv",
    output_video="flux_roi_overlay.mp4",
)
```

**ROI dimensions must be exact multiples of the patch size** (width divisible by 160, height divisible by 128) to ensure the tiled patch grid fills the ROI without remainder.

### Output DataFrame Columns

| Column | Description |
|---|---|
| `frame_idx` | Frame number |
| `timestamp_s` | Time in seconds |
| `frame_count` | Total density sum (instantaneous bat count) |
| `flux_per_frame` | Net bats crossing the counting line this frame |
| `cumulative_flux` | Running total — final value is the population estimate |

The output video shows the ROI and a side-by-side density overlay with the counting line drawn, flow arrows indicating direction, and live count/flux/total annotations.

---

## File Reference

| File / Directory | Description |
|---|---|
| `CPI_CSRNet.ipynb` | Main notebook — model definition, training, inference, flux |
| `csrnet_meshed_best.pth` | Best model checkpoint (saved during training) |
| `training_history.csv` | Per-epoch loss, MAE, RMSE log |
| `video/output.mp4` | Input thermal video |
| `video/meshed_output/output/v1/output_counted.mp4` | Annotated output video |
| `video/meshed_output/output/v1/frame_counts.csv` | Per-frame count log |
| `video/meshed_output/output/v1/count_over_time.png` | Count-over-time plot |
| `flux_roi_results.csv` | Per-frame flux and cumulative population estimate |
| `flux_roi_overlay.mp4` | Flux visualization video |
| `logs/missing_maps` | JSONL log of images skipped due to missing density files |

---

## Dependencies

```
torch
torchvision
numpy
opencv-python (cv2)
Pillow
matplotlib
pandas
```

Runs on **Google Colab** with GPU. All paths are relative to `MyDrive/CPI_Folder_CSRNet/`.

---

## Design Decisions

**Why density estimation instead of detection/tracking?** Individual bat detection fails in dense emergence swarms due to severe occlusion and motion blur. Density maps aggregate spatial bat distribution without requiring one-to-one correspondences across frames.

**Why CSRNet?** CSRNet was originally developed for crowd counting in dense human scenes — structurally analogous to dense bat emergence — and its dilated backend preserves spatial resolution while capturing multi-scale context.

**Why sliding window inference?** The model was trained on 128×128 patches. Sliding window inference keeps input distribution consistent with training, avoids resolution mismatch on arbitrary-size frames, and allows the full-frame density map to be assembled from patch-level predictions.

**Why flux instead of peak count?** Peak instantaneous count underestimates population because not all bats are in frame simultaneously. Optical flow flux integrated over the full emergence event gives a more accurate total colony size estimate.

---

## NSF Center for Pandemic Insights

This system is designed for long-term passive wildlife monitoring. Bat colonies are reservoirs for zoonotic viruses (SARS-CoV, MERS-CoV, Ebola, Nipah). Automated colony size tracking over seasons enables detection of anomalous population changes that may correlate with disease dynamics — supporting early warning infrastructure without requiring physical disturbance of roost sites.
