# BCDetection

Bottle cap detection project using YOLOv8 for three classes:

- `closed_cap`
- `half_cap`
- `no_cap`

This repository contains the dataset used for training, the scripts for auto-labeling and training, the testing scripts, and the final report PDF.

## Repository Structure

```text
BCDetection/
  data/
    auto_label.py
    data_labeled_roboflow/
      data.yaml
      train/
      valid/
    for_test/
  train/
    hyperparam_compare.py
    augmentation_compare.py
  test/
    predict.py
    test_augmentation_weights.py
  Bottle_Cap_Detection_Final_Report.pdf
  README.md
```

## Dataset

The final YOLOv8 dataset is stored in:

```text
data/data_labeled_roboflow/
```

This dataset contains three bottle-cap classes:

- `closed_cap`: bottle cap is fully closed
- `half_cap`: bottle cap is partially open or loose
- `no_cap`: bottle has no cap

Dataset split:

| Split | Description |
| --- | --- |
| `train/` | Training images and YOLO labels |
| `valid/` | Validation images and YOLO labels |

External test images used for manual evaluation are stored in:

```text
data/for_test/
```

## Source Files

| File | Purpose |
| --- | --- |
| `data/auto_label.py` | Uses Grounding DINO to create initial YOLO-format labels automatically. |
| `train/hyperparam_compare.py` | Compares batch size, epoch, and learning rate settings. |
| `train/augmentation_compare.py` | Compares YOLO augmentation policies. |
| `test/predict.py` | Runs inference on images, folders, or webcam using the selected final model. |
| `test/test_augmentation_weights.py` | Tests all augmentation weights on the same external image set. |

## Method Summary

1. Images were collected manually by taking bottle photos in three cap states.
2. Initial bounding boxes were generated with Grounding DINO using class-specific prompts.
3. The labels were checked and corrected manually in Roboflow.
4. YOLOv8n was trained and compared across multiple hyperparameters:
   - batch size
   - epoch count
   - learning rate
5. The selected training setup was then used to compare YOLO augmentation strategies.
6. The final model was tested on external images and compared with a closed-up-only dataset experiment.

## Final Selected Training Setup

The final selected full-project model used:

```text
Model: YOLOv8n
Batch size: 8
Epochs: 150
Learning rate: 0.005
Optimizer: SGD
Augmentation: No Mosaic
```

## Running the Scripts

Auto-labeling:

```powershell
python data/auto_label.py
```

Hyperparameter comparison:

```powershell
python train/hyperparam_compare.py
```

Augmentation comparison:

```powershell
python train/augmentation_compare.py
```

Prediction on test images:

```powershell
python test/predict.py --source data/for_test
```

## Report

The final report submitted for the project is included as:

```text
Bottle_Cap_Detection_Final_Report.pdf
```
