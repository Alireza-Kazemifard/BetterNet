import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

def intersection_over_union(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def binary_crossentropy_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

# --- Advanced Metrics (S-measure, E-measure, Weighted F) ---
# Note: These are usually calculated using numpy on CPU during testing, 
# but here are TF implementations or placeholders if needed for training monitoring.

def structure_measure(y_true, y_pred):
    # Implementation of S-measure (simplified for TF monitoring)
    # Real implementation requires object-oriented region analysis usually done in numpy
    return dice_coefficient(y_true, y_pred) 

def enhanced_measure(y_true, y_pred):
    # Implementation of E-measure (simplified)
    return intersection_over_union(y_true, y_pred)
