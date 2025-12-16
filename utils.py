
import os
import tensorflow as tf
from metrics import *

def create_directory(path):
    if not os.path.exists(path): os.makedirs(path)

def load_model(model_path):
    custom_objects = {
        'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 
        'bce_dice_loss': bce_dice_loss, 'binary_crossentropy_dice_loss': bce_dice_loss
    }
    try:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)
    except:
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
