# Road Vehicle Detection and Tracking

本项目使用 Kaggle 的 **Road Vehicle Images Dataset** 微调 YOLOv8n，并完成道路车辆检测、视频多目标跟踪、遮挡 / ID 跳变分析和越线计数。

## 项目链接
* Model Weights: [https://drive.google.com/file/d/1zcZLa_ImniVXMuKyDiwSHHv4ebbTCT6V/view?usp=sharing](https://drive.google.com/file/d/1zcZLa_ImniVXMuKyDiwSHHv4ebbTCT6V/view?usp=sharing)
* Dataset: [https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset](https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset)

## 1. 环境配置

```bash
conda create -n vehicle_tracking python=3.10 -y
conda activate vehicle_tracking
pip install -r requirements.txt
```

登录 W&B：

```bash
wandb login
```

## 2. 数据集下载

使用 Kaggle API：

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
python scripts/download_dataset.py --out datasets/road_vehicle
```

检查数据集：

```bash
python scripts/check_dataset.py --data configs/road_vehicle.yaml
```

数据集目录示例：

```text
datasets/road_vehicle/trafic_data/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

## 3. 模型训练

```bash
yolo detect train \
  model=./yolov8n.pt \
  data=configs/road_vehicle.yaml \
  epochs=80 \
  imgsz=640 \
  batch=16 \
  optimizer=SGD \
  lr0=0.01 \
  device=0 \
  project=/dev/shm/rv_work/runs \
  name=yolov8n-road-vehicle \
  amp=False
```

## 4. 模型验证

```bash
yolo detect val \
  model=/dev/shm/rv_work/runs/yolov8n-road-vehicle-3/weights/best.pt \
  data=configs/road_vehicle.yaml \
  imgsz=640 \
  batch=16 \
  device=0
```

## 5. 视频多目标跟踪

```bash
python scripts/track_video.py \
  --weights /dev/shm/rv_work/runs/yolov8n-road-vehicle-3/weights/best.pt \
  --source videos/block.mp4 \
  --output results/final/tracking_output_botsort_final.mp4 \
  --tracker botsort.yaml
```

可选：ByteTrack。

```bash
python scripts/track_video.py \
  --weights /dev/shm/rv_work/runs/yolov8n-road-vehicle-3/weights/best.pt \
  --source videos/block.mp4 \
  --output results/final/tracking_output_bytetrack_final.mp4 \
  --tracker bytetrack.yaml
```

## 6. 越线计数

```bash
python scripts/line_count.py \
  --weights /dev/shm/rv_work/runs/yolov8n-road-vehicle-3/weights/best.pt \
  --source videos/block.mp4 \
  --output results/final/line_count_final.mp4 \
  --tracker botsort.yaml \
  --line 960 0 960 1080 #根据你的视频大小改变
```


## 7. 项目结构

```text
road_vehicle_yolov8_tracking/
├── configs/
│   └── road_vehicle.yaml
├── scripts/
│   ├── check_dataset.py
│   ├── download_dataset.py
│   ├── extract_occlusion_frames.py
│   ├── line_count.py
│   ├── plot_results_local.py
│   ├── track_video.py
│   ├── train.py
│   └── val.py
├── docs/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. 说明

仓库中不包含完整数据集、训练权重和大视频原件。数据集请从 Kaggle 下载，训练好的模型权重见上方 Google Drive 链接。
