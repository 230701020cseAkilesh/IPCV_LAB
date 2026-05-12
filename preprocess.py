import cv2

def preprocess(img):
    """
    Three-stage preprocessing pipeline:
    1. Resize  - standardise to 640x480
    2. Denoise - Gaussian blur 3x3
    3. Enhance - histogram equalisation on grayscale
    Returns grayscale enhanced image.
    """
    img  = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    img  = cv2.GaussianBlur(img, (3, 3), 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray
