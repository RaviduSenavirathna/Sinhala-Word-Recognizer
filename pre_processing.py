import cv2
import matplotlib.pyplot as plt

# To perform pre-processing on an image, including converting to grayscale, applying Gaussian blur, and performing Canny edge detection.
def img_processing(img):
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # smooth (important for pencil)
    img = cv2.GaussianBlur(img, (5, 5), 0)


    # Canny edge detection
    processed_img = cv2.Canny(
        img,
        threshold1=60,
        threshold2=60
    )

    return processed_img


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