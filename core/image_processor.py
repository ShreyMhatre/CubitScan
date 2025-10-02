import cv2
import numpy as np

def get_best_contrast_channel(image):
    """
    Analyzes the R, G, and B channels of an image and returns the one
    with the best contrast for marker detection.

    Args:
        image (np.array): The raw input BGR image.

    Returns:
        np.array: The single-channel grayscale image with the best contrast.
    """
    print("Analyzing color channels for best contrast...")
    
    b, g, r = cv2.split(image)
    
    # ============================ FIX IS HERE ============================
    # Corrected the typo from COLOR_BGR_GRAY to COLOR_BGR2GRAY
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # =====================================================================

    channels = {'blue': b, 'green': g, 'red': r, 'gray': gray}
    max_contrast = 0
    best_channel_name = 'gray'
    best_channel_image = gray

    h, w = image.shape[:2]
    roi = (slice(int(h*0.25), int(h*0.75)), slice(int(w*0.25), int(w*0.75)))

    for name, channel_img in channels.items():
        contrast = cv2.Laplacian(channel_img[roi], cv2.CV_64F).var()
        if contrast > max_contrast:
            max_contrast = contrast
            best_channel_name = name
            best_channel_image = channel_img
            
    print(f" -> Best contrast found in '{best_channel_name}' channel.")
    return best_channel_image

def preprocess_with_clahe(image):
    """
    Applies CLAHE to a BGR image.
    """
    print("Applying CLAHE pre-processing for detection...")

    # ============================ FIX IS HERE ============================
    # Corrected the typo from COLOR_BGR_GRAY to COLOR_BGR2GRAY
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # =====================================================================

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    return enhanced_gray