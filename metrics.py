import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

smooth = 1e-15

@register_keras_serializable(package="builtins")
def intersection_over_union(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Use reshape instead of Flatten layer for Keras 3 compatibility
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable(package="builtins")
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@register_keras_serializable(package="builtins")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

@register_keras_serializable(package="builtins")
def binary_crossentropy_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

@register_keras_serializable(package="builtins")
def weighted_f_score(y_true, y_pred, beta=2):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    true_positive = tf.reduce_sum(y_true_f * y_pred_f)
    false_positive = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    false_negative = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    
    precision = true_positive / (true_positive + false_positive + smooth)
    recall = true_positive / (true_positive + false_negative + smooth)
    
    f_score = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall + smooth)
    return f_score

@register_keras_serializable(package="builtins")
def s_score(y_true, y_pred, alpha=0.5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Simplified for monitoring (Full S-measure requires CPU/Numpy)
    return dice_coefficient(y_true, y_pred)

@register_keras_serializable(package="builtins")
def e_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Simplified for monitoring
    return intersection_over_union(y_true, y_pred)

@register_keras_serializable(package="builtins")
def mean_absolute_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs(y_pred - y_true))
