# Report Checklist

## Must-have content

- [ ] Model structure: YOLOv8 single-stage detector, backbone/neck/head overview.
- [ ] Dataset: Kaggle Road Vehicle Images Dataset, train/valid split, 21 classes.
- [ ] Experimental settings:
  - [ ] train/valid split
  - [ ] model name
  - [ ] image size
  - [ ] batch size
  - [ ] learning rate
  - [ ] optimizer
  - [ ] epochs
  - [ ] loss function: box loss, cls loss, DFL loss
  - [ ] metrics: precision, recall, mAP@0.5, mAP@0.5:0.95
- [ ] W&B screenshots:
  - [ ] train/box_loss, train/cls_loss, train/dfl_loss
  - [ ] val/box_loss, val/cls_loss, val/dfl_loss
  - [ ] metrics/precision(B), metrics/recall(B)
  - [ ] metrics/mAP50(B), metrics/mAP50-95(B)
- [ ] Video tracking output screenshot with bbox + class + tracking ID.
- [ ] Occlusion / dense interaction analysis with 3-4 consecutive frames.
- [ ] Line-crossing counting explanation and result screenshot.
- [ ] GitHub repo URL.
- [ ] Model weights cloud drive URL.

## Suggested hyperparameter table

| Item | Value |
|---|---|
| Dataset | Road Vehicle Images Dataset |
| Train / validation split | original train / valid split |
| Model | YOLOv8n |
| Pretrained weights | yolov8n.pt |
| Image size | 640 |
| Batch size | 16 |
| Epochs | 80 |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| Weight decay | 0.0005 |
| Loss | box loss + cls loss + DFL loss |
| Metrics | Precision, Recall, mAP@0.5, mAP@0.5:0.95 |
| Tracker | BoT-SORT or ByteTrack |
