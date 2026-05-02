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
- Example:

```python
from pre_processing import processed_img_preview

processed_img_preview('dataset/ක/handwriting1.png')
```

Display Details:
- Figure size: 4×4 inches
- Colormap: Grayscale
- Axes: Hidden for cleaner view
- Title: "Processed Image"