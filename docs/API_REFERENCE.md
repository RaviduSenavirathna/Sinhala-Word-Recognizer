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

