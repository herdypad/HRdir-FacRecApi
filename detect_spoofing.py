import numpy as np
import cv2
def detect_spoofing(file):
    """
    Detect whether the image is a spoof or real.
    """
    # Check if the input is a file object or a file path
    if hasattr(file, 'read'):
        # Read the image data from the file object
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        # Read the image from the file path
        image = cv2.imread(file)

    if image is None:
        raise ValueError("Invalid image file")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Threshold to determine if the image is a spoof or real
    if laplacian_var < 100:
        return True  # Spoof
    else:
        return False  # Real