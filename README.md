# Road Vehicle Detection, Tracking, and Line-Crossing Counting

This project fine-tunes YOLOv8 on the Kaggle **Road Vehicle Images Dataset** and applies the trained detector to video multi-object tracking, occlusion / ID-switch analysis, and line-crossing vehicle counting.

Dataset page:
https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset

## 1. Environment

```bash
conda create -n vehicle_tracking python=3.10 -y
conda activate vehicle_tracking
pip install -r requirements.txt
```

Log in to Weights & Biases before training:

```bash
wandb login
```

## 2. Download Dataset

Create a Kaggle API token from your Kaggle account settings, then put `kaggle.json` at `~/.kaggle/kaggle.json`.

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
python scripts/download_dataset.py --out datasets/road_vehicle
```

Expected dataset structure:

```text
datasets/road_vehicle/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

If your extracted folders differ, update `configs/road_vehicle.yaml` accordingly.

Check the dataset:

```bash
python scripts/check_dataset.py --data configs/road_vehicle.yaml
```

## 3. Train YOLOv8 with W&B

```bash
python scripts/train.py \
  --data configs/road_vehicle.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --optimizer SGD \
  --lr0 0.01 \
  --device 0 \
  --wandb \
  --wandb-project road-vehicle-yolov8-tracking \
  --name yolov8n-road-vehicle
```

Training outputs:

```text
runs/vehicle_detection/yolov8n-road-vehicle/
├── results.csv
├── results.png
└── weights/
    ├── best.pt
    └── last.pt
```

The W&B run will contain training loss curves, validation loss curves, precision, recall, and mAP curves. Use screenshots from W&B in the PDF report.

## 4. Validate

```bash
python scripts/val.py \
  --weights runs/vehicle_detection/yolov8n-road-vehicle/weights/best.pt \
  --data configs/road_vehicle.yaml \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --output-json results/val_metrics.json
```

## 5. Video Tracking

Put a 10-30 second test video under `videos/test_video.mp4`, then run:

```bash
python scripts/track_video.py \
  --weights runs/vehicle_detection/yolov8n-road-vehicle/weights/best.pt \
  --source videos/test_video.mp4 \
  --output results/tracking_output.mp4 \
  --tracker botsort.yaml \
  --conf 0.25 \
  --iou 0.5
```

You can also try ByteTrack:

```bash
python scripts/track_video.py \
  --weights runs/vehicle_detection/yolov8n-road-vehicle/weights/best.pt \
  --source videos/test_video.mp4 \
  --output results/tracking_output_bytetrack.mp4 \
  --tracker bytetrack.yaml
```

## 6. Line-Crossing Counting

Adjust the counting line coordinates to your video frame. Format: `X1 Y1 X2 Y2`.

```bash
python scripts/line_count.py \
  --weights runs/vehicle_detection/yolov8n-road-vehicle/weights/best.pt \
  --source videos/test_video.mp4 \
  --output results/line_count_output.mp4 \
  --tracker botsort.yaml \
  --line 200 450 1100 450 \
  --save-frames 120 121 122 123
```

Outputs:

```text
results/line_count_output.mp4
results/crossing_events.csv
results/occlusion_frames/frame_000120.jpg
results/occlusion_frames/frame_000121.jpg
results/occlusion_frames/frame_000122.jpg
results/occlusion_frames/frame_000123.jpg
```

## 7. Extract Occlusion Frames Manually

If you already have the annotated video and want to extract frames 120-123:

```bash
python scripts/extract_occlusion_frames.py \
  --source results/line_count_output.mp4 \
  --start 120 \
  --num 4 \
  --out results/occlusion_frames
```

## 8. Optional Local Plots

The report requirement asks for W&B or SwanLab screenshots. The following local plots are only backup figures.

```bash
python scripts/plot_results_local.py \
  --results-csv runs/vehicle_detection/yolov8n-road-vehicle/results.csv \
  --out-dir results/local_plots
```

## 9. Model Weights Link

Upload this file to Google Drive / Baidu Netdisk:

```text
runs/vehicle_detection/yolov8n-road-vehicle/weights/best.pt
```

Then paste the public download link into the PDF report.

## 10. Required Links in Final PDF Report

- GitHub repo: `https://github.com/<your-name>/road-vehicle-detection-tracking`
- Model weights: Google Drive / Baidu Netdisk link
- W&B screenshots: training loss, validation loss, precision/recall/mAP curves
