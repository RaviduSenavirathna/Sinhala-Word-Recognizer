# Step-by-Step Tutorial

Complete walkthrough of the Sinhala Word Recognizer pipeline with examples and explanations.

## Table of Contents

1. [Preparing Your Dataset](#preparing-your-dataset)
2. [Image Preprocessing](#image-preprocessing)
3. [Character Segmentation](#character-segmentation)
4. [Model Training](#model-training)
5. [Making Predictions](#making-predictions)
6. [Tips & Optimization](#tips--optimization)

---

## Preparing Your Dataset

### Understanding the Dataset Structure

Your dataset needs to follow a specific folder structure:
```
dataset/ 
├── ක/ # First Sinhala character 
│   ├── 001.png 
│   ├── 002.png 
│   ├── 003.png 
│   └── ... 
├── ග/ # Second character 
│   ├── 001.png 
│   ├── 002.png 
│   └── ... 
├── ත/ 
├── න/ 
├── ප/ 
├── ම/ 
├── ර/ 
├── ල/ 
├── ස/ 
└── හ/ # Last character 
└── ...
```


**Key Points:**
- Each folder name is the Sinhala character itself
- Each folder contains images of that character
- Images can be PNG, JPG, or other common formats
- More samples = better model accuracy

### Recommended Dataset Guidelines

| Aspect | Recommendation |
|--------|-----------------|
| Samples per character | 100-500 minimum |
| Image quality | Clear, scanned/photographed handwriting |
| Image size | 400×400 or larger |
| Balance | Similar count across all characters |
| File format | PNG or JPG |

### Loading Your Dataset

**Step 1:** Upload to Google Colab (if using Colab)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset to Colab
import shutil
shutil.copytree('/content/drive/MyDrive/dataset', '/content/dataset')
```

**Step 2:** Verify dataset in dataset.ipynb
```python
import os

dataset_path = 'dataset'
classes = os.listdir(dataset_path)

print(f"Total characters: {len(classes)}")
print(f"Characters: {sorted(classes)}\n")

# Count samples per character
for char in sorted(classes):
    count = len(os.listdir(os.path.join(dataset_path, char)))
    print(f"{char}: {count} samples")
```

**Expected Output:**
```
Total characters: 10
Characters: ['ක', 'ග', 'ත', 'න', 'ප', 'ම', 'ර', 'ල', 'ස', 'හ']

ක: 150 samples
ග: 145 samples
ත: 152 samples
...
```
