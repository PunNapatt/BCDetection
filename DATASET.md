# Dataset Information

## Dataset Collection

The dataset was collected manually by taking photos of water bottles in three cap-status classes:

- `closed_cap`: bottle cap is fully closed
- `half_cap`: bottle cap is partially closed or loose
- `no_cap`: bottle has no cap

The first dataset focused on closed-up cap images. After testing, more half-bottle and full-bottle images were collected to make the final model work better on practical test images.

## Auto-Labeling and Manual Checking

Initial bounding boxes were generated using Grounding DINO with class-specific prompts:

| Class | Prompt | Threshold |
| --- | --- | --- |
| `closed_cap` | `bottle cap . closed bottle cap . sealed cap .` | `0.30` |
| `half_cap` | `bottle cap . partially open cap . loose cap . half open bottle cap .` | `0.28` |
| `no_cap` | `bottle . open bottle . bottle neck . bottle without cap .` | `0.28` |

The auto-labeled boxes were manually checked and corrected in Roboflow, then exported in YOLOv8 format.

## Dataset Paths

Original class-folder dataset:

```text
DataPreparation/DATA bottle_cap/
```

Final YOLOv8 dataset:

```text
Trainandrun/Data label from roboflow/
```

External test images:

```text
DataPreparation/For_Test/
```

## Final YOLOv8 Dataset Structure

```text
Trainandrun/Data label from roboflow/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
```

The final YOLOv8 dataset contains:

| Split | Images | Labels |
| --- | ---: | ---: |
| Train | 260 | 260 |
| Validation | 64 | 64 |
| Total | 324 | 324 |

## Git Hosting Note

The dataset folders contain hundreds of image files and may be large for normal Git hosting.

Recommended options:

- Use Git LFS for image files and model weights.
- Or upload the dataset to Google Drive/Roboflow and replace this section with the dataset link.

Dataset link:

```text
[Add Google Drive / Roboflow dataset link here if the dataset is not committed to Git]
```
