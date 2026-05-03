# API Reference

Complete documentation for all functions available in the Sinhala Word Recognizer project.

## pre_processing.py

#### `img_processing(img)`

Converts a raw image to a binary thresholded image suitable for character segmentation and classification.

**Parameters:**
- `img` (numpy.ndarray): Input image loaded via cv2.imread() (BGR format)

**Returns:**
- `thresh` (numpy.ndarray): Binary thresholded image (0 and 255 values only)

**Process:**
1. Converts BGR image to grayscale
2. Applies Gaussian blur (3×3 kernel) to reduce noise
3. Applies Otsu's threshold for automatic optimal thresholding
4. Inverts binary image (foreground becomes white, background black)

**Example:**
```python
import cv2
from pre_processing import img_processing

# Load image
img = cv2.imread('handwritten_text.png')

# Process image
processed = img_processing(img)

# Display result
cv2.imshow('Processed', processed)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Technical Details:
- Uses cv2.COLOR_BGR2GRAY for color conversion
- Gaussian blur kernel: (3, 3) with sigma = 0
- Threshold method: cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
- Output range: [0, 255] (binary: 0 or 255)



#### `tight_crop(img)`
Crops the image tightly around non-zero pixels to remove excess white space.

**Parameters:**
- `img` (numpy.ndarray): Binary image from img_processing()

**Returns:**
- `cropped` (numpy.ndarray): Tightly cropped image with minimal padding

**Behavior:**
1. Finds all non-zero pixel coordinates
2. Crops to bounding box with 2-3 pixel padding
3. Returns original image if no non-zero pixels found

**Example:**
```python
from pre_processing import img_processing, tight_crop

processed = img_processing(img)
cropped = tight_crop(processed)
```

Use Cases:
- Remove extra whitespace after segmentation
- Normalize character size before resizing
- Prepare characters for 128×128 resizing


#### `processed_img_preview(img_path)`
Loads an image, applies preprocessing, and displays the result using Matplotlib.

**Parameters:**
- img_path (str): File path to the image (e.g., 'dataset/ක/sample.png')

**Returns:**
- None (displays plot)

**Raises:**
- ValueError: If image file not found at path

Example:
```python
from pre_processing import processed_img_preview

processed_img_preview('dataset/ක/handwriting1.png')
```

Display Details:
- Figure size: 4×4 inches
- Colormap: Grayscale
- Axes: Hidden for cleaner view
- Title: "Processed Image"


# Notebook Functions

## dataset.ipynb
Main functions available in the dataset preparation notebook:

**Load and Explore Dataset**
```python
# Lists all character classes
classes = os.listdir('dataset')
print(f"Found {len(classes)} character classes")

# Shows sample count per character
for char in classes:
    count = len(os.listdir(f'dataset/{char}'))
    print(f"{char}: {count} samples")
```

**Create Processed Dataset**
Applies preprocessing to all images and saves to `dataset_processed/`
- Input: Raw images from `dataset/`
- Output: Processed binary images in `dataset_processed/`
- Processing: Uses `img_processing()` function

**Data Visualization**
```python
# Display sample images from each class
# Shows original and processed versions side-by-side
```


## word_segmenting.ipynb
Character segmentation and standardization functions.

**Vertical Projection Segmentation**
Splits multi-character words into individual characters using vertical projection.

```python
# Key Process:
# 1. Calculate vertical projection (sum of white pixels per column)
# 2. Find valleys (low projection values) = character boundaries
# 3. Extract character bounding boxes
# 4. Resize to 128×128
```

**Image Resizing & Padding**
```python
# Resizes character to 128×128 while maintaining aspect ratio
# Adds padding to maintain shape integrity
```
**Output**
- Directory: `segmented_dataset/`
- Format: Individual character images, 128×128 pixels
- Each character organized by class



## model_training.ipynb
Deep learning model training functions.

**Model Architecture**
```python
# CNN Architecture:
model = tf.keras.Sequential([
    # Input layer: 128×128×1 (grayscale)
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    # Output: Number of Sinhala characters
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

**Training**
```python
# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train on segmented dataset
history = model.fit(
    train_data,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)
```

**Model Saving**
```python
# Save trained weights
model.save('sinhala_model.h5')
```

**Hyperparameters:**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Learning rate: 0.001 (default)
- Batch size: 32
- Epochs: 50 (adjustable)
- Validation split: 0.2 (20% validation data)


## predict.ipynb
Inference and prediction functions.

**Load Model**
```python
import tensorflow as tf

model = tf.keras.models.load_model('sinhala_model.h5')
```

**Single Image Prediction**
```python
def predict_character(img_path, model):
    """
    Predict Sinhala character from image
    
    Parameters:
    - img_path: Path to test image
    - model: Loaded TensorFlow model
    
    Returns:
    - character: Predicted Sinhala character
    - confidence: Prediction confidence (0-1)
    """
    # Load and preprocess image
    img = cv2.imread(img_path)
    processed = img_processing(img)
    cropped = tight_crop(processed)
    
    # Resize to 128×128
    resized = cv2.resize(cropped, (128, 128))
    
    # Normalize
    normalized = resized / 255.0
    
    # Add batch dimension
    batch = np.expand_dims(normalized, axis=[0, -1])
    
    # Predict
    prediction = model.predict(batch)
    
    return character_class, confidence
```

**Batch Prediction**

```python
def predict_word(word_image_path, model):
    """
    Recognize full word from image
    
    Process:
    1. Preprocess image
    2. Segment characters
    3. Predict each character
    4. Combine predictions
    """
```

### Constants & Configuration
**Image Dimensions**
- Standard size: 128×128 pixels
- Color space: Grayscale (1 channel)
- Normalization: [0, 1] (divide by 255)

**Processing Parameters**
- Gaussian blur kernel: (3, 3)
- Threshold method: Otsu's (automatic)
- Binary inversion: Yes (white foreground)

**Model Parameters**
- Input shape: (128, 128, 1)
- Output activation: Softmax (multiclass)
- Training optimizer: Adam
- Learning rate: 0.001 (default)

**Error Codes & Exceptions**
|   Error   |   Cause   |	Solution    |
| --------- | --------- | ------------- |
|   ValueError: Image not found    |	Invalid file path   |	Check file path and extension   |
|   OOM (Out of Memory) |   Batch size too large    |	Reduce batch_size in config |
|   Shape mismatch  |	Input not 128×128   |	Verify segmentation step    |
|   Low accuracy    |	Insufficient training data  |	Collect more samples    |



### Best Practices
Always preprocess before segmentation

```python
processed = img_processing(img)
cropped = tight_crop(processed)
```

Maintain 128×128 standard
- Don't change input dimensions after model training
- Resize with aspect ratio preservation

Use validation split during training
- Prevents overfitting
- Monitor validation accuracy

Save model checkpoints
```python
model.save(f'checkpoint_epoch_{epoch}.h5')
```

Test on diverse data
- Different writing styles
- Various image qualities
- Different character sizes


## Version Information
- TensorFlow: 2.10+
- OpenCV: 4.5+
- NumPy: 1.20+
- Python: 3.7+

For more information, see the main README.md or GETTING_STARTED.md.