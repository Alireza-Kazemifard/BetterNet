import os
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import (intersection_over_union, dice_coefficient, dice_loss, 
                     binary_crossentropy_dice_loss, weighted_f_score, 
                     s_score, e_score, mean_absolute_error)

def create_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
    except OSError:
        print(f"Error: creating directory with name {directory_path}")

def load_trained_model(model_path):
    with CustomObjectScope({
            'intersection_over_union': intersection_over_union,
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'binary_crossentropy_dice_loss': binary_crossentropy_dice_loss,
            'weighted_f_score': weighted_f_score,
            's_score': s_score,
            'e_score': e_score,
            'mean_absolute_error': mean_absolute_error
        }):
        loaded_model = tf.keras.models.load_model(model_path)
        return loaded_model
