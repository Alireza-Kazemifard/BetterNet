import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

smooth = 1e-15

@register_keras_serializable(package="builtins")
def intersection_over_union(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable(package="builtins")
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

@register_keras_serializable(package="builtins")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

@register_keras_serializable(package="builtins")
def binary_crossentropy_dice_loss(y_true, y_pred):
    # ترکیب BCE و Dice طبق مقاله
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

@register_keras_serializable(package="builtins")
def weighted_f_score(y_true, y_pred, beta=2):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_positive = tf.reduce_sum(y_true * y_pred)
    false_positive = tf.reduce_sum((1 - y_true) * y_pred)
    false_negative = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = true_positive / (true_positive + false_positive + smooth)
    recall = true_positive / (true_positive + false_negative + smooth)
    
    f_score = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + smooth)
    return f_score

@register_keras_serializable(package="builtins")
def s_score(y_true, y_pred, alpha=0.5):
    # پیاده‌سازی سازگار با TF برای مانیتورینگ
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_positive = tf.reduce_sum(y_true * y_pred)
    false_positive = tf.reduce_sum((1 - y_true) * y_pred)
    false_negative = tf.reduce_sum(y_true * (1 - y_pred))
    
    s_object = true_positive / (true_positive + false_negative + smooth)
    s_region = true_positive / (true_positive + false_positive + smooth)
    return alpha * s_object + (1 - alpha) * s_region

@register_keras_serializable(package="builtins")
def e_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_positive = tf.reduce_sum(y_true * y_pred)
    false_positive = tf.reduce_sum((1 - y_true) * y_pred)
    false_negative = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = true_positive / (true_positive + false_positive + smooth)
    recall = true_positive / (true_positive + false_negative + smooth)
    
    return 2 * precision * recall / (precision + recall + smooth)

@register_keras_serializable(package="builtins")
def mean_absolute_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs(y_pred - y_true))
