# Troubleshooting Guide

Solutions for common issues and error messages.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data & Preprocessing Issues](#data--preprocessing-issues)
- [Segmentation Issues](#segmentation-issues)
- [Training Issues](#training-issues)
- [Prediction Issues](#prediction-issues)
- [Performance Issues](#performance-issues)

---

## Installation Issues

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Error Message:**
```code
ModuleNotFoundError: No module named 'tensorflow'
```

**Causes:**
- Dependencies not installed
- Wrong Python version
- Pip installation failed

**Solutions:**

1. **Reinstall all dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Install individually:**
    ```bash
    pip install tensorflow>=2.10
    pip install opencv-python>=4.5
    pip install numpy>=1.20
    pip install matplotlib>=3.4
    pip install jupyter
    ```

3. **Check Python version:**
    ```bash
    python --version  # Should be 3.7 or higher
    ```

4. **Use virtual environment (recommended):**
    ```bash 
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

5. **For Google Colab (no installation needed):**
    ```bash
    # Run in first cell
    !pip install -q tensorflow opencv-python
    ```


### Issue: "ImportError: cannot import name 'cv2'"

**Causes:**
- OpenCV not installed or wrong version
- Using old cv2 API

**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-python==4.7.0.72
```


### Issue: Pip takes too long / Won't install

**Solution - Use Conda instead:**
```bash
conda create -n sinhala python=3.9
conda activate sinhala
conda install tensorflow opencv numpy matplotlib jupyter
```


# Data & Preprocessing Issues

### Issue: "File not found" or "Cannot read image"

**Error Message:**
```code
ValueError: Image not found
```

**Causes:**
- Wrong file path
- File doesn't exist
- Encoding issues

**Solutions:**

1. Check file path:
    ```python
    import os

    # Verify path exists
    path = 'dataset/ක/image.png'
    print(os.path.exists(path))  # Should print: True
    ```

2. List available files:
    ```python
    import os

    dataset_path = 'dataset'
    for char in os.listdir(dataset_path):
        char_path = os.path.join(dataset_path, char)
        files = os.listdir(char_path)
        print(f"{char}: {len(files)} files")
    ```

3. Use absolute path:
    ```python
    import os

    img_path = os.path.abspath('dataset/ක/image.png')
    img = cv2.imread(img_path)
    ```

4. For Colab - mount drive first:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')

    img_path = '/content/drive/MyDrive/dataset/ක/image.png'
    img = cv2.imread(img_path)
    ```

### Issue: Dataset folder structure incorrect

**Problem:**
```code
dataset/
├── character1.png           ❌ Wrong: images in root
├── character2.png
└── character3.png
```

**Solution:**
```code
dataset/
├── ක/                      ✓ Correct: folders per character
│   ├── image1.png
│   └── image2.png
└── ග/
    └── image1.png
```

**Script to reorganize:**
```python
import os
import shutil
from pathlib import Path

# Create proper structure
dataset_path = 'dataset_new'
Path(dataset_path).mkdir(exist_ok=True)

# Example: create character folders
characters = ['ක', 'ග', 'ත', 'න', 'ප', 'ම', 'ර', 'ල', 'ස', 'හ']
for char in characters:
    Path(os.path.join(dataset_path, char)).mkdir(exist_ok=True)

# Move/copy your images into appropriate folders
```


### Issue: Preprocessed images look bad (too dark/light, blurry)

**Problem:**
- Segmentation fails on bad preprocessed images
- Low accuracy later

**Solutions:**
1. Adjust Gaussian blur:
    ```python 
    # In pre_processing.py, try different kernel sizes
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Larger = more blur
    # or
    blur = cv2.GaussianBlur(gray, (3, 3), 1)  # Sigma helps too
    ```

2. Adjust threshold:
    ```python
    # Try manual threshold instead of Otsu
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    # Adjust 127 based on image brightness
    ```

3. Check input images:
    - Are originals clear and visible?
    - High contrast (dark text on light background)?
    - Reasonable resolution (400×400+)?

4. Visualize preprocessing:
    ```python
    from pre_processing import processed_img_preview

    # Check several samples
    processed_img_preview('dataset/ක/001.png')
    processed_img_preview('dataset/ක/002.png')
    processed_img_preview('dataset/ក/003.png')
    ```

### Issue: "NoneType" error when reading image

**Error Message:**
```code
AttributeError: 'NoneType' object has no attribute 'shape'
```

**Cause:** `cv2.imread() returned None (couldn't read file)`

**Solution:**
```python
import cv2

img_path = 'dataset/ක/image.png'
img = cv2.imread(img_path)

# Check if image loaded successfully
if img is None:
    print(f"Error: Could not read {img_path}")
    print(f"File exists: {os.path.exists(img_path)}")
else:
    print(f"Image shape: {img.shape}")
```


# Segmentation Issues

### Issue: Over-segmentation (too many character fragments)

**Problem:**
- One character split into multiple pieces
- Extra segments created

**Example:**
```code 
Word: "ක" (one character)
Result: [piece1, piece2, piece3]  ❌ Should be 1
```

**Solution - Increase threshold:**

In `word_segmenting.ipynb`, find the line:
```python
threshold = 5  # Current value
```

Try larger values:
```python
threshold = 10  # Try this
# or
threshold = 15  # Or this
```

**Test threshold:**
```python 
def test_threshold(img, threshold):
    projection = np.sum(img > 0, axis=0)
    valleys = np.sum(projection < threshold)
    print(f"Threshold {threshold}: {valleys} valleys found")

img = cv2.imread('dataset_processed/ක/001.png', cv2.IMREAD_GRAYSCALE)
for t in [3, 5, 10, 15, 20]:
    test_threshold(img, t)
```


### Issue: Under-segmentation (characters merged together)

**Problem:**
- Multiple characters merged into one segment
- Missing characters in output

**Example:**
```code 
Word: "ක" + "ග" (two characters)
Result: [merged_piece]  ❌ Should be 2
```

**Solution - Decrease threshold:**
```python
threshold = 2  # Try smaller value
# or
threshold = 1  # Even smaller
```


### Issue: Segmented characters have wrong size

**Problem:**
- Resize creates distorted characters
- Characters stretched or compressed

**Solution - Maintain aspect ratio:**
```python
# Current (might distort):
char_resized = cv2.resize(char_img, (128, 128))

# Better (maintains aspect ratio):
def resize_with_aspect_ratio(img, size=128):
    h, w = img.shape
    aspect = w / h
    
    if aspect > 1:  # Wider than tall
        new_w = size
        new_h = int(size / aspect)
    else:  # Taller than wide
        new_h = size
        new_w = int(size * aspect)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    # Pad to 128×128
    canvas = np.zeros((size, size), dtype=np.uint8)
    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas
```


# Training Issues

### Issue: "Out of Memory" (OOM) error

**Error Message:**
```code
tensorflow.python.framework.errors_impl.ResourceExhaustedError: 
OOM when allocating tensor with shape [batch_size, ...]
```

**Causes:**
- Batch size too large for available memory
- GPU memory full (if using GPU)
- Dataset too large

**Solutions:**
1. **Reduce batch size:**
    ```python
    # Current in model_training.ipynb
    history = model.fit(
        X_train, y_train,
        batch_size=32,  # Try reducing
        ...
    )

    # Try:
    batch_size=16  # or 8
    ```
2. **Use Google Colab instead:**
- Free GPU with more memory
- Enable GPU: Runtime → Change runtime type → GPU

3. **Reduce dataset size:**
    ```python 
    # Use only a subset for testing
    X_train = X_train[:1000]  # First 1000 samples
    y_train = y_train[:1000]
    ```

4. **Clear memory:**
    ```python
    import gc
    gc.collect()  # Force garbage collection

    # Or in Colab:
    !nvidia-smi  # Check GPU memory
    ```

### Issue: Extremely slow training (hours for 1 epoch)

**Problem:**
- Training on CPU (very slow)
- Dataset too large
- GPU not being used

**Solutions:**

1. **Use GPU (Colab):**
    ```python
    # Check GPU availability
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))

    # Output if GPU available:
    # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
    ```

2. **Enable GPU in Colab:**
    - Runtime → Change runtime type → Hardware accelerator → GPU

3. **Reduce dataset:**
    ```python 
    # Use subset for testing
    X_train = X_train[:10000]  # Use 10k samples
    y_train = y_train[:10000]
    ```

4. **Reduce model complexity:**
    ```python
    # Simpler model (fewer layers/filters)
    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    ```


### Issue: Accuracy not improving (stuck at low value)

**Problem:**
- Training loss doesn't decrease
- Validation accuracy stays at random levels (~10% for 10 classes)
- Model not learning

**Solutions:**
1. **Check data loading:**
    ```python
    # Verify labels are one-hot encoded
    print(y_train_encoded[0])  # Should have one 1 and rest 0s

    # Verify image normalization
    print(np.min(X_train), np.max(X_train))  # Should be 0-1
    ```

2. **Verify model compiles correctly:**
    ```python
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    ```

3. **Check if data is shuffled:**
    ```python
    # If not shuffled, model might see same labels repeatedly
    from sklearn.utils import shuffle
    X_train, y_train = shuffle(X_train, y_train)
    ```

4. **Use learning rate decay:**
    ```python
    import tensorflow as tf

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.96
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    ```


