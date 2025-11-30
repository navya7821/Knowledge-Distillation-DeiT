# DeiT-Small Knowledge Distillation Project

## Project Overview
This project demonstrates knowledge distillation for a binary image classification task (classes 0 and 1) using a fine-tuned ResNet-50 as the teacher model and DeiT-Small (from timm library) as the student model. The teacher is pre-trained on ImageNet and fine-tuned on a custom dataset with 2 classes, achieving high validation accuracy. The student is trained via distillation on the same dataset. The dataset appears imbalanced based on test supports (21 samples for class 0, 279 for class 1 in the test set). Testing is performed on 300 unseen images to evaluate true generalization, with no data leakage. The notebook (`DeiT_Small.ipynb`) includes unzipping data, teacher fine-tuning, student training (distillation), and evaluation.

## Key Parameters
- **Teacher Model**: ResNet-50 (pre-trained on ImageNet1K_V1), fine-tuned to 2 output classes.
- **Student Model**: DeiT-Small (distilled variant with 2 output classes).
- **Training Epochs**: 50 (for teacher fine-tuning; student training likely similar based on code structure).
- **Batch Size**: 32.
- **Optimizer**: SGD (learning rate=0.01, momentum=0.9, weight decay=1e-4).
- **Scheduler**: CosineAnnealingLR (T_max=50).
- **Loss Function**: CrossEntropyLoss (for teacher); distillation-specific logic in student training (hard labels visible in test).
- **Image Processing**: Input size 224x224; augmentations include RandomResizedCrop, RandomHorizontalFlip, ColorJitter; normalization with ImageNet means/std.
- **Dataset Paths**: Train/Val/Test directories under `/content/Dataset` and `/content/test_set`; labels from CSV files.
- **Hardware**: GPU (CUDA-enabled); libraries include PyTorch, torchvision, timm, pandas, sklearn, matplotlib.
- **Evaluation Metrics**: Accuracy, balanced accuracy, recall per class, macro F1, confusion matrix.

## Numeric Comparison of Soft and Hard Distillation
Test set: 300 unseen images (21 class 0, 279 class 1).  
All metrics are from the final test evaluation.

| Metric                | Soft Distillation | Hard Distillation |
|-----------------------|-------------------|-------------------|
| Accuracy              | 0.9300            | 0.9300            |
| Balanced Accuracy     | 0.5000            | 0.5220            |
| Recall (Class 0)      | 0.0000            | 0.0476            |
| Recall (Class 1)      | 1.0000            | 0.9964            |
| Macro F1-Score        | 0.4800            | 0.5253            |

Both distillation approaches achieve identical overall accuracy, but hard distillation shows marginal improvement on the minority class (class 0) due to slightly better recall. The imbalance heavily favors class 1 performance. Confusion matrices and full reports are in the notebook.
