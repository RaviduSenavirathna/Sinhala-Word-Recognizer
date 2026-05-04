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

# Image Preprocessing

## Why Preprocessing Matters

Raw images have:
- Noise: Dust, scanning artifacts
- Variations: Brightness, contrast differences
- Complexity: Color information not needed for shapes

**Preprocessing converts raw images into clean binary images** suitable for character recognition.

### The Preprocessing Pipeline
```
Raw Image (BGR, colored, noisy)
    ↓
1. Convert to Grayscale (simplify to single channel)
    ↓
2. Apply Gaussian Blur (reduce noise)
    ↓
3. Apply Otsu Threshold (convert to pure black & white)
    ↓
Processed Image (binary, clean)
```

### Step-by-Step Preprocessing
Open `dataset.ipynb` and follow this workflow:

**Step 1: Load and Preview Raw Image**

```python
import cv2
import matplotlib.pyplot as plt
from pre_processing import processed_img_preview

# View a sample raw image
img_path = 'dataset/ක/001.png'
img = cv2.imread(img_path)

plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Raw Image")
plt.axis('off')
plt.show()
```

**Step 2: Preprocess Single Image**

```python
from pre_processing import img_processing, tight_crop

# Preprocess
processed = img_processing(img)

# Crop excess white space
cropped = tight_crop(processed)

# Display result
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cropped, cmap='gray')
plt.title("Processed")
plt.axis('off')
plt.show()
```

**Step 3: Batch Process All Dataset Images**

```python
import os
import numpy as np
from pathlib import Path

source_dir = 'dataset'
target_dir = 'dataset_processed'

# Create output directory
Path(target_dir).mkdir(exist_ok=True)

# Process all images
for character in os.listdir(source_dir):
    char_source = os.path.join(source_dir, character)
    char_target = os.path.join(target_dir, character)
    
    # Create character folder
    Path(char_target).mkdir(exist_ok=True)
    
    # Process each image
    for img_file in os.listdir(char_source):
        try:
            img_path = os.path.join(char_source, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Skipped: {img_path} (read error)")
                continue
            
            # Preprocess
            processed = img_processing(img)
            
            # Save
            output_path = os.path.join(char_target, img_file)
            cv2.imwrite(output_path, processed)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("✓ Preprocessing complete! Saved to dataset_processed/")
```



### Understanding Each Processing Step

1. Grayscale Conversion

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
- Converts 3-channel (BGR) to 1-channel (grayscale)
- Keeps only brightness information
- Reduces data size by 3×

2. Gaussian Blur

```python
blur = cv2.GaussianBlur(gray, (3, 3), 0)
```
- Smooths image to reduce noise
- Kernel size (3, 3) = small neighborhood
- Sigma = 0 = auto-calculated
- Effect: Fuzzy the image slightly

3. Otsu Threshold

```python
_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
```
- Converts grayscale → pure black/white
- `THRESH_OTSU`: Automatically finds best threshold value
- `THRESH_BINARY_INV`: Inverts (makes text white, background black)
- Result: Clean binary image perfect for segmentation



### Preprocessing Results
You should see:
- Before: Noisy, colored, variable contrast
- After: Clean, binary (only 0 and 255 values), consistent


# Character Segmentation

### What is Segmentation?
**Problem**: Your dataset contains full words/lines, but you need individual characters.

**Solution**: Automatically split words into characters using Vertical Projection.

## Vertical Projection Method

```
Word Image: "ශ්‍ර"
    ↓
Calculate: Sum of white pixels in each column
    ↓
Find valleys: Columns with few/no pixels
    ↓
Identify boundaries: Gaps between characters
    ↓
Extract regions: Individual character bounding boxes
    ↓
Resize: All to 128×128 pixels
    ↓
Output: Individual character images
```

### Implementing Segmentation

Open `word_segmenting.ipynb` and follow these steps:

**Step 1: Load Preprocessed Image**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a preprocessed word image
img = cv2.imread('dataset_processed/ක/001.png', cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray')
plt.title("Input: Preprocessed Word")
plt.show()
```

**Step 2: Calculate Vertical Projection**
```python
# Sum white pixels in each column
vertical_projection = np.sum(img > 0, axis=0)

# Plot projection
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title("Word Image")

plt.subplot(1, 2, 2)
plt.plot(vertical_projection)
plt.title("Vertical Projection")
plt.xlabel("Column")
plt.ylabel("White Pixels")
plt.show()

# Output example:
# [0, 0, 15, 23, 18, 12, 0, 0, 5, 20, 22, ...]
#                       ↑ Valley = boundary
```

**Step 3: Find Character Boundaries**
```python
# Find columns with no pixels (valleys)
threshold = 5  # Minimum pixels to consider "character"
valleys = np.where(vertical_projection < threshold)[0]

# Find contiguous regions of valleys
boundaries = []
in_valley = False

for i in range(len(vertical_projection)):
    is_valley = vertical_projection[i] < threshold
    
    if is_valley and not in_valley:
        boundaries.append(('start', i))
        in_valley = True
    elif not is_valley and in_valley:
        boundaries.append(('end', i))
        in_valley = False

print("Character boundaries found at columns:", boundaries)
```

**Step 4: Extract and Resize Characters**
```python
def segment_word_to_characters(img_path, output_dir):
    """
    Segment a word image into individual characters
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Vertical projection
    projection = np.sum(img > 0, axis=0)
    
    # Find valleys
    threshold = 5
    boundaries = []
    in_valley = False
    
    for i in range(len(projection)):
        is_valley = projection[i] < threshold
        
        if is_valley and not in_valley:
            boundaries.append(i)
            in_valley = True
        elif not is_valley and in_valley:
            boundaries.append(i)
            in_valley = False
    
    # Extract characters
    characters = []
    for j in range(0, len(boundaries)-1, 2):
        start = boundaries[j]
        end = boundaries[j+1]
        
        # Extract region
        char_img = img[:, start:end]
        
        # Resize to 128×128 maintaining aspect ratio
        char_resized = cv2.resize(char_img, (128, 128))
        
        characters.append(char_resized)
    
    return characters

# Usage
chars = segment_word_to_characters('dataset_processed/ක/001.png', 'segmented_dataset')

# Visualize segmented characters
plt.figure(figsize=(15, 2))
for i, char in enumerate(chars):
    plt.subplot(1, len(chars), i+1)
    plt.imshow(char, cmap='gray')
    plt.axis('off')
plt.suptitle("Segmented Characters")
plt.show()
```

**Step 5: Batch Segment All Images**
```python
import os
from pathlib import Path

source_dir = 'dataset_processed'
target_dir = 'segmented_dataset'

Path(target_dir).mkdir(exist_ok=True)

# Track statistics
stats = {}

for character in os.listdir(source_dir):
    char_source = os.path.join(source_dir, character)
    char_target = os.path.join(target_dir, character)
    
    Path(char_target).mkdir(exist_ok=True)
    
    count = 0
    
    for img_file in os.listdir(char_source):
        try:
            img_path = os.path.join(char_source, img_file)
            chars = segment_word_to_characters(img_path, char_target)
            
            # Save each segmented character
            for idx, char_img in enumerate(chars):
                output_name = f"{img_file[:-4]}_{idx}.png"
                output_path = os.path.join(char_target, output_name)
                cv2.imwrite(output_path, char_img)
                count += 1
        
        except Exception as e:
            print(f"Error: {img_path} - {e}")
    
    stats[character] = count
    print(f"✓ {character}: {count} characters segmented")

print(f"\nTotal characters segmented: {sum(stats.values())}")
```


### Segmentation Troubleshooting

**Problem:** Too many false segments (over-segmentation)
- **Solution:** Increase threshold value (e.g., 10 instead of 5)

**Problem:** Characters grouped together (under-segmentation)
- **Solution:** Decrease threshold value

**Problem:** Missing characters
- **Solution:** Check if preprocessing was effective; characters might be too faint




# Model Training

### Setting Up Training

Open `model_training.ipynb`:

**Step 1: Load Segmented Dataset**
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from pathlib import Path

# Load dataset
dataset_dir = 'segmented_dataset'
classes = sorted(os.listdir(dataset_dir))

print(f"Classes found: {classes}")
print(f"Total classes: {len(classes)}\n")

# Load images and labels
X = []  # Images
y = []  # Labels (class indices)

for class_idx, character in enumerate(classes):
    char_dir = os.path.join(dataset_dir, character)
    
    for img_file in os.listdir(char_dir):
        try:
            img_path = os.path.join(char_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Normalize to [0, 1]
            img = img / 255.0
            
            # Add channel dimension (needed for Conv2D)
            img = np.expand_dims(img, axis=-1)
            
            X.append(img)
            y.append(class_idx)
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Dataset loaded:")
print(f"  X shape: {X.shape}  # (samples, height, width, channels)")
print(f"  y shape: {y.shape}  # (samples,)")
print(f"  Unique classes: {len(np.unique(y))}")
```

**Step 2: Split into Train/Validation**
```python
from sklearn.model_selection import train_test_split

# Split 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# One-hot encode labels (required for categorical_crossentropy)
from tensorflow.keras.utils import to_categorical
y_train_encoded = to_categorical(y_train, num_classes=len(classes))
y_val_encoded = to_categorical(y_val, num_classes=len(classes))
```

**Step 3: Build CNN Model**
```python
model = keras.Sequential([
    # Block 1
    keras.layers.Conv2D(32, (3, 3), activation='relu', 
                       input_shape=(128, 128, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Block 2
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Block 3
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    
    # Dense layers
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),  # Prevent overfitting
    
    # Output layer
    keras.layers.Dense(len(classes), activation='softmax')
])

model.summary()
```

Output:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 126, 126, 32)     320
 max_pooling2d (MaxPooling2D (None, 63, 63, 32)       0
 conv2d_1 (Conv2D)           (None, 61, 61, 64)       18496
 max_pooling2d_1 (MaxPooling(None, 30, 30, 64)       0
 ...
 dense_1 (Dense)             (None, 256)              33280
 dropout (Dropout)           (None, 256)              0
 dense_2 (Dense)             (None, 10)               2570
=================================================================
Total params: 328,842
```

**Step 4: Compile Model**
```python
model.compile(
    optimizer='adam',           # Learning algorithm
    loss='categorical_crossentropy',  # Loss function
    metrics=['accuracy']         # Monitor accuracy
)
```

**Step 5: Train Model**
```python
# Train on GPU (Colab) or CPU
history = model.fit(
    X_train, y_train_encoded,
    epochs=50,                   # Number of iterations through data
    batch_size=32,              # Samples per gradient update
    validation_data=(X_val, y_val_encoded),
    verbose=1                   # Print progress
)

# Output example:
# Epoch 1/50
# 45/45 [==============================] - 12s 270ms/step - loss: 2.3891 - accuracy: 0.1241 - val_loss: 2.2956 - val_accuracy: 0.1543
# Epoch 2/50
# 45/45 [==============================] - 10s 220ms/step - loss: 2.1893 - accuracy: 0.2156 - val_loss: 1.8976 - val_accuracy: 0.3821
# ...
```

**Step 6: Visualize Training Progress**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Print final metrics
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
```

**Step 7: Save Trained Model**
```python
# Save model weights
model.save('sinhala_model.h5')

# Or save in newer format
model.save('sinhala_model')

print("✓ Model saved successfully!")
```




# Making Predictions

### Using Trained Model

Open `predict.ipynb`:

**Step 1: Load Model**
```python
import tensorflow as tf
import cv2
import numpy as np

# Load trained model
model = tf.keras.models.load_model('sinhala_model.h5')

# Load class names
classes = sorted(os.listdir('segmented_dataset'))
print(f"Classes: {classes}")
```

**Step 2: Predict Single Image**
```python
def predict_character(img_path):
    """
    Predict Sinhala character from single image
    """
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Preprocess (same as training)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    # Predict
    prediction = model.predict(img_batch, verbose=0)
    
    # Get class with highest probability
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    character = classes[class_idx]
    
    return character, confidence, prediction[0]

# Test
test_img = 'segmented_dataset/ක/001_0.png'
char, conf, probs = predict_character(test_img)

print(f"Predicted: {char}")
print(f"Confidence: {conf:.2%}")
print(f"Top 3 predictions:")
for idx in np.argsort(probs)[-3:][::-1]:
    print(f"  {classes[idx]}: {probs[idx]:.2%}")
```

**Step 3: Predict Multiple Images**
```python
def predict_directory(directory_path):
    """
    Predict characters from all images in directory
    """
    results = []
    
    for img_file in os.listdir(directory_path):
        try:
            img_path = os.path.join(directory_path, img_file)
            char, conf, _ = predict_character(img_path)
            results.append({
                'file': img_file,
                'prediction': char,
                'confidence': conf
            })
        except Exception as e:
            print(f"Error predicting {img_file}: {e}")
    
    return results

# Test on a directory
test_results = predict_directory('test_images')

# Display results
for result in test_results[:10]:  # Show first 10
    print(f"{result['file']}: {result['prediction']} ({result['confidence']:.2%})")
```

**Step 4: Evaluate on Test Set**
```python
# Load test data
X_test = []
y_test = []

# ... (load images)

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred_classes == y_test)
print(f"Test Accuracy: {accuracy:.2%}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix
import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```




# Tips & Optimization

### Training Optimization

1. Use GPU
- Google Colab: Free T4 GPU
- Training 10-50× faster than CPU
- Enable in Colab: Runtime → Change runtime type → GPU

2. Batch Size
- Larger batch = faster training, more memory
- Smaller batch = slower training, more accurate
- Recommended: 16-64

3. Learning Rate
```python 
optimizer = keras.optimizers.Adam(learning_rate=0.001) 
```

4. Epochs
- Too few: Underfitting (low accuracy)
- Too many: Overfitting (high train accuracy, low test accuracy)
- Use early stopping:

```python
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop if no improvement for 5 epochs
    restore_best_weights=True
)
```

### Improving Accuracy

1. Data Quality
- Clean, consistent images
- Balanced classes (similar samples per character)
- Remove blurry/damaged images

2. Data Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2
)
```

3. Model Architecture
- More layers = more capacity but longer training
- More filters = better feature extraction
- Dropout = prevent overfitting

4. Preprocessing
- Ensure clean segmentation
- Consistent image sizes
- Proper normalization

### Deployment Tips

1. Convert to Mobile Format
```python
converter = tf.lite.TFLiteConverter.from_saved_model('sinhala_model')
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

2. Save Class Mapping
```python
import json

class_mapping = {idx: char for idx, char in enumerate(classes)}

with open('classes.json', 'w', encoding='utf-8') as f:
    json.dump(class_mapping, f, ensure_ascii=False)
```

3. Create Inference Pipeline
```python
class SinhalaRecognizer:
    def __init__(self, model_path, classes_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path) as f:
            self.classes = json.load(f)
    
    def recognize(self, image_path):
        # Load → Preprocess → Segment → Predict
        pass
```