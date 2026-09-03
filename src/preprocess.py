import cv2
import numpy as np
import torch

def apply_clahe(image_np):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image_np.shape) == 3:
        channels = cv2.split(image_np)
        eq_channels = [clahe.apply(c) for c in channels]
        return cv2.merge(eq_channels)
    return clahe.apply(image_np)

def z_score_normalize(image_tensor):
    mean = image_tensor.mean()
    std = image_tensor.std()
    if std == 0:
        return image_tensor
    return (image_tensor - mean) / std