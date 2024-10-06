# feature_extraction.py

import cv2
import numpy as np

def extract_features(image):
    """Extract features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges.flatten()  # Flatten the 2D array to a 1D array for training

def extract_features_from_frame(frame):
    """Extract features from a video frame."""
    return extract_features(frame)  # Reuse the existing feature extraction function
