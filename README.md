readme = """# Cricket Run-Out Detection

A deep learning system that classifies cricket run-out decisions — `out` vs `not_out` —
from images and video footage using transfer learning and computer vision.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Inference Results](#inference-results)
- [Video Detection Results](#video-detection-results)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Author](#author)

---

## Overview

Run-out decisions in cricket are traditionally reviewed manually by third umpires
frame by frame. This project explores whether a convolutional neural network can
learn to make the same `out` / `not_out` call directly from a single frame or a
short video clip, using transfer learning on top of ImageNet-pretrained backbones.

## Problem Statement

Given an image or video frame of a run-out appeal, predict whether the batsman is:

- `out` — bat / body has not reached the crease before the bails are dislodged
- `not_out` — bat / body is grounded safely before the bails are dislodged

This is framed as a binary image classification problem, with the video use case
treated as repeated frame-level classification aggregated into a final decision.

---

## Project Structure

\`\`\`
cricket_runout_detection/
├── data/
│   ├── raw/              → 225 original images (out: 127, not_out: 98)
│   ├── data_augmented/   → 1800 balanced images (900 per class)
│   └── split/            → train / val / test splits
├── models/
│   ├── best_model.keras          → MobileNetV2
│   └── best_model_resnet.keras   → ResNet50V2
├── results/               → all charts and evaluation images
├── videos/
│   ├── input/             → test videos (out / not_out)
│   └── output/            → annotated output videos
└── notebooks/
    ├── 01_data_preparation_eda.ipynb
    ├── 02_model_training_evaluation.ipynb
    ├── 03_model_inference_prediction.ipynb
    ├── 04_video_runout_detection.ipynb
    └── 05_final_report_comparison.ipynb
\`\`\`

---

## Dataset

- 225 raw cricket images collected manually from match footage and review clips
- 2 classes: `out` (127 images) and `not_out` (98 images)
- augmented to 1800 images (900 per class) using 9 augmentation types
  (rotation, flip, brightness, zoom, shift, shear, channel shift)
- split 70 / 15 / 15 → train (1260), validation (270), test (270), stratified by class

---

## Methodology

1. **Data preparation** — collected and labeled raw images, checked class balance
2. **Augmentation** — balanced classes, expanded dataset to reduce overfitting
3. **Transfer learning** — fine-tuned MobileNetV2 and ResNet50V2 with a custom head
4. **Evaluation** — compared both models on accuracy, F1 score, confusion matrix
5. **Static inference** — ran the winning model on unseen test images
6. **Video inference** — extracted frames with OpenCV, classified each frame,
   aggregated into a single video-level decision

---

## Models

| Model       | Base     | Total Layers | Test Accuracy | F1 Score | Errors |
|-------------|----------|--------------|----------------|----------|--------|
| MobileNetV2 | ImageNet | 159          | 93.00%         | 0.93     | 20/270 |
| ResNet50V2  | ImageNet | 195          | 95.00%         | 0.95     | 14/270 |

**Winner: ResNet50V2** — higher test accuracy, higher F1 score, fewer errors.

### Training Curves Comparison
![Training Curves](results/training_curves_comparison.png)

### Confusion Matrix Comparison
![Confusion Matrix](results/confusion_matrix_comparison.png)

---

## Inference Results

### Batch Predictions on Test Images
![Batch Predictions](results/batch_predictions.png)

---

## Video Detection Results

Frame-by-frame prediction was run on real cricket match videos using the
ResNet50V2 model. Each frame was classified independently, and the final
video-level decision was taken as the majority prediction across frames.

| Video      | True Label | Out Frames | Not Out Frames | Decision | Correct |
|------------|------------|------------|-----------------|----------|---------|
| Project 19 | out        | 149        | 0               | out      | ✓       |
| Project 18 | not_out    | 0          | 125             | not_out  | ✓       |

- avg confidence out video     : 99.79%
- avg confidence not_out video : 84.96%
- both test videos classified correctly at the video level

---

## Tech Stack

- **Language**: Python 3.11
- **Deep Learning**: TensorFlow, Keras (MobileNetV2, ResNet50V2 transfer learning)
- **Computer Vision**: OpenCV (video frame extraction and inference)
- **Data Handling**: NumPy
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: scikit-learn
- **Environment**: Jupyter Notebook

## Requirements

\`\`\`
tensorflow==2.21.0
keras==3.14.1
opencv-python==4.13.0
numpy==2.4.1
matplotlib==3.10.9
seaborn==0.13.2
pillow==12.2.0
scikit-learn
\`\`\`

---

## How to Run

1. clone the repository
2. install requirements: `pip install -r requirements.txt`
3. run notebooks in order from 01 to 05
4. trained models are saved to `models/`, all charts to `results/`
5. to test on a new video, place it in `videos/input/` and run notebook 04

---

## Results Summary

- raw dataset        : 225 images
- augmented dataset  : 1800 images
- final model        : ResNet50V2
- test accuracy      : 95.00%
- macro f1 score     : 0.95
- video detection    : 100% correct on both test videos

---

## Limitations

- trained mainly on review-style, close-up footage; accuracy drops on
  wide-angle broadcast footage due to domain gap
- small raw dataset (225 images) before augmentation limits diversity of
  camera angles, lighting, and pitch conditions
- video decision uses majority frame voting, not temporal modeling, so it
  does not use motion information across frames

## Future Work

- fine-tune on broadcast-angle footage to close the domain gap
- add Grad-CAM visualizations for model explainability
- explore temporal models (3D CNN / LSTM over frame sequences) for
  video-level classification instead of frame-majority voting
- expand raw dataset with more varied match conditions and camera angles

---

## Author

**Shiva**
GitHub: [shiva5019](https://github.com/shiva5019)
"""

readme_path = Path(r"C:\Users\J.Shiva\OneDrive\Attachments\cricket_runout_detection\README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

print(f"README saved to : {readme_path}")
