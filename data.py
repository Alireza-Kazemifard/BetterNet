import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from glob import glob

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def load_data(dataset_paths, split=0.1):
    images = []
    masks = []

    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    print("📊 Loading data paths...")
    for path in dataset_paths:
        # جستجوی فایل‌ها با پسوندهای مختلف
        curr_images = sorted(glob(os.path.join(path, "images", "*")))
        curr_masks = sorted(glob(os.path.join(path, "masks", "*")))
        
        # فیلتر کردن پسوندهای مجاز
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        curr_images = [x for x in curr_images if x.lower().endswith(valid_exts)]
        curr_masks = [x for x in curr_masks if x.lower().endswith(valid_exts)]
        
        # اطمینان از برابر بودن تعداد
        min_len = min(len(curr_images), len(curr_masks))
        curr_images = curr_images[:min_len]
        curr_masks = curr_masks[:min_len]

        if len(curr_images) > 0:
             print(f"   Found {len(curr_images)} pairs in {path}")
             images.extend(curr_images)
             masks.extend(curr_masks)

    # شافل کردن داده‌ها
    train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=split, random_state=42)
    return (train_x, train_y), (valid_x, valid_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    # طبق مقاله و کد قدیم: تغییر سایز با Lanczos4
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    # اصلاح حیاتی: نرمال‌سازی به 0 تا 1
    x = x / 255.0
    return x.astype(np.float32)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    # نرمال‌سازی ماسک
    x = x / 255.0
    x = np.expand_dims(x, axis=-1)
    # باینری کردن ماسک
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

def tf_dataset(X, Y, batch_size=8, augment=False, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
        
    if augment:
        # داده‌افزایی ساده (Flipping) مشابه مقاله
        dataset = dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), tf.image.random_flip_left_right(y)))
        dataset = dataset.map(lambda x, y: (tf.image.random_flip_up_down(x), tf.image.random_flip_up_down(y)))
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
