import tensorflow as tf
from keras.saving import register_keras_serializable

SMOOTH = 1e-7


@register_keras_serializable()
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + SMOOTH) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + SMOOTH)


@register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


@register_keras_serializable()
def intersection_over_union(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)


@register_keras_serializable()
def mean_absolute_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.reduce_mean(tf.abs(y_true - y_pred))


@register_keras_serializable()
def weighted_f_score(y_true, y_pred, beta=0.3):
    # simple weighted F-beta (approx; paper uses Fan toolbox)
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + SMOOTH)
    recall = tp / (tp + fn + SMOOTH)
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + SMOOTH)


@register_keras_serializable()
def s_score(y_true, y_pred):
    # lightweight proxy (NOT the full Fan toolbox S-measure)
    # keeps behavior closer to old repo than the simplified new version
    return dice_coefficient(y_true, y_pred)


@register_keras_serializable()
def e_score(y_true, y_pred):
    # lightweight proxy (NOT the full Fan toolbox E-measure)
    return intersection_over_union(y_true, y_pred)


@register_keras_serializable()
def max_e_score(y_true, y_pred):
    # sweep thresholds to approximate maxE
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    thresholds = tf.linspace(0.0, 1.0, 21)
    best = tf.constant(0.0, dtype=tf.float32)

    for t in tf.unstack(thresholds):
        yp = tf.cast(y_pred > t, tf.float32)
        best = tf.maximum(best, intersection_over_union(y_true, yp))

    return best


@register_keras_serializable()
def binary_crossentropy_dice_loss(y_true, y_pred, alpha=0.5):
    # matches paper description: L = alpha*BCE + (1-alpha)*DiceLoss
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    dl = dice_loss(y_true, y_pred)
    return alpha * bce + (1.0 - alpha) * dl
