import cv2
import matplotlib.pyplot as plt

# To perform pre-processing on an image, including converting to grayscale, applying Gaussian blur, and performing Canny edge detection.
def img_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold (simpler & stable)
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return thresh


def processed_img_crop(img):

    thresh = img_processing(img)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = thresh[y:y+h, x:x+w]
    else:
        cropped = thresh

    return cropped

# To load an image, perform pre-processing, and display the Canny edge image using Matplotlib.
def processed_img_preview(img_path):
    # ---- load image ----
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Image not found")

    processed_img = img_processing(img)

    plt.figure(figsize=(4,4))
    plt.title("Processed Image")
    plt.imshow(processed_img, cmap="gray")
    plt.axis("off")

    plt.show()