import cv2


# To perform pre-processing on an image, including converting to grayscale, applying Gaussian blur, and performing Canny edge detection.
def img_processing(img):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # smooth (important for pencil)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)


    # Canny edge detection
    edges = cv2.Canny(
        gray,
        threshold1=60,
        threshold2=60
    )

    return edges