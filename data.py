import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

def load_data(dataset_path, split=0.1):
    images = sorted([os.path.join(dataset_path, "images", x) for x in os.listdir(os.path.join(dataset_path, "images"))])
    masks = sorted([os.path.join(dataset_path, "masks", x) for x in os.listdir(os.path.join(dataset_path, "masks"))])

    train_x, test_x, train_y, test_y = train_test_split(images, masks, test_size=split, random_state=42)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=split, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT))
    x = x / 255.0
    return x.astype(np.float32)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT))
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    x = np.where(x > 0.5, 1.0, 0.0)
    return x.astype(np.float32)

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    y.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    return x, y

def tf_dataset(X, Y, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    return dataset
