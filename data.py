import os
import cv2
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split

def load_data(path, split=0.1, seed=42):
    """
    Loads image and mask paths and splits them into train/val using random shuffle.
    """
    images = sorted(glob(os.path.join(path, "images", "*")))
    masks = sorted(glob(os.path.join(path, "masks", "*")))
    
    # Use random split to avoid bias (better than taking last 10%)
    train_x, valid_x, train_y, valid_y = train_test_split(
        images, masks, test_size=split, random_state=seed
    )
    
    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    return x.astype(np.float32)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    # TRAINING OPTIMIZATION:
    # Use standard resize (LINEAR) to keep soft edges.
    # Do NOT binarize here (remove > 127).
    # This helps the loss function (BCE+Dice) propagate gradients better.
    x = cv2.resize(x, (256, 256))
    x = x / 255.0  # Normalized to [0, 1]
    
    x = np.expand_dims(x, axis=-1)
    return x.astype(np.float32)

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
