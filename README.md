# Bottle Cap Detection Using YOLOv8

This project detects water-bottle cap status in three classes:

- `closed_cap`
- `half_cap`
- `no_cap`

The workflow includes manual image collection, Grounding DINO assisted auto-labeling, Roboflow manual checking, YOLOv8 training, hyperparameter comparison, augmentation comparison, and external image testing.

## Important Source Files

| File | Purpose |
| --- | --- |
| `DataPreparation/auto_label.py` | Auto-label images with Grounding DINO prompts and export YOLO-format labels. |
| `Trainandrun/train_yolo.py` | Baseline YOLOv8 training script. |
| `Trainandrun/hyperparam_compare.py` | Batch size, epoch, and learning-rate comparison script. |
| `Trainandrun/augmentation_compare.py` | YOLO augmentation policy comparison script. |
| `Trainandrun/test_augmentation_weights.py` | Tests each augmentation weight on the same external test images. |
| `Trainandrun/predict.py` | Inference script for images, folders, or webcam. |
| `PROJECT_STRUCTURE.md` | Project folder map. |
| `DATASET.md` | Dataset description and dataset submission notes. |

## Dataset

The main YOLOv8 dataset is stored in:

```text
Trainandrun/Data label from roboflow/
```

The original class-folder dataset used before Roboflow export is stored in:

```text
DataPreparation/DATA bottle_cap/
```

See `DATASET.md` for details.

## Final Selected Model

The selected model configuration is:

```text
Model: YOLOv8n
Batch size: 8
Epochs: 150
Learning rate: 0.005
Optimizer: SGD
Augmentation: No Mosaic
```

Final selected weight:

```text
Trainandrun/yolo_augmentation_output/aug_compare_20260512_181523/no_mosaic/weights/best.pt
```

## How to Run Prediction

```powershell
& "C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe" "Trainandrun\predict.py" --source "DataPreparation\For_Test" --conf 0.4
```

For webcam:

```powershell
& "C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe" "Trainandrun\predict.py" --source webcam --conf 0.4
```

## Git Submission Notes

This project contains many images and `.pt` model files, so Git LFS is recommended.

```powershell
git lfs install
git init
git add .gitattributes .gitignore README.md DATASET.md PROJECT_STRUCTURE.md
git add DataPreparation Trainandrun
git commit -m "Add bottle cap detection project source and dataset"
```

If the dataset is too large for your Git hosting limit, upload the dataset to Google Drive/Roboflow and put the link in `DATASET.md`.
