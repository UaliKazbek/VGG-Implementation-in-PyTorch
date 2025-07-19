# VGG-Implementation-in-PyTorch
This repository contains a clean and modular PyTorch implementation of the **VGG** architecture, including variants VGG-11, VGG-13, VGG-16, and VGG-19.

##  Supported Models

-  VGG-11
-  VGG-13
-  VGG-16
-  VGG-19

Each model follows the official layer configuration from the paper.

---

##  Model Architecture

Example: **VGG-11**
``` text
Input (3, 224, 224)
→ [Conv2d → ReLU] ×1
→ MaxPool
→ [Conv2d → ReLU] ×1
→ MaxPool
→ [Conv2d → ReLU] ×2
→ MaxPool
→ [Conv2d → ReLU] ×2
→ MaxPool
→ [Conv2d → ReLU] ×2
→ MaxPool
→ Flatten
→ [Linear → ReLU → Dropout] ×2
→ Linear → num_classes
```
## Features
- Custom configuration for each VGG variant
- Modular architecture with make_layers() function
- Uses nn.Sequential for clean code
- Easily extendable for other tasks (e.g., fine-tuning, feature extraction)
