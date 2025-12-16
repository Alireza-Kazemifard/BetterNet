import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, Activation, BatchNormalization, UpSampling2D,
    Input, Concatenate, Add, Dropout, GlobalAveragePooling2D,
    Reshape, Dense, GlobalMaxPooling2D, Lambda, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.regularizers import l1_l2


def channel_attention_module(input_feature, ratio=8):
    # safer channel dim
    channel = tf.keras.backend.int_shape(input_feature)[-1]
    if channel is None:
        raise ValueError("Channel dimension is None. Please provide a fully-defined input shape.")

    shared_dense_one = Dense(
        channel // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_dense_two = Dense(
        channel,
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )

    # (B, C)
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    # (B, C)
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    # (B, C)
    attention = Add()([avg_pool, max_pool])
    attention = Activation("sigmoid")(attention)

    # IMPORTANT: reshape to (B, 1, 1, C) so multiply is correct
    attention = Reshape((1, 1, channel))(attention)

    return Multiply()([input_feature, attention])


def spatial_attention_module(input_feature):
    kernel_size = 7
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    attention = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(concat)

    return Multiply()([input_feature, attention])


def cbam_module(input_feature, ratio=8):
    x = channel_attention_module(input_feature, ratio)
    x = spatial_attention_module(x)
    return x


def squeeze_excitation_module(input_feature, ratio=16):
    filters = tf.keras.backend.int_shape(input_feature)[-1]
    if filters is None:
        raise ValueError("Channel dimension is None. Please provide a fully-defined input shape.")

    se_shape = (1, 1, filters)

    squeeze = GlobalAveragePooling2D()(input_feature)  # (B, C)
    squeeze = Reshape(se_shape)(squeeze)               # (B, 1, 1, C)

    excitation = Dense(filters // ratio, activation="relu")(squeeze)
    excitation = Dense(filters, activation="sigmoid")(excitation)

    return Multiply()([input_feature, excitation])


def residual_block(input_feature, num_filters, dropout_rate=0.5):
    x = input_feature

    x = Conv2D(num_filters // 4, (1, 1), padding="same",
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters // 4, (3, 3), padding="same",
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same",
               kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(num_filters, (1, 1), padding="same",
                      kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input_feature)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    x = squeeze_excitation_module(x)
    x = Dropout(dropout_rate)(x)
    return x


def BetterNet(input_shape=(224, 224, 3), num_classes=1, dropout_rate=0.5, freeze_encoder=True):
    inputs = Input(input_shape)

    # IMPORTANT:
    # data.py outputs images in range [0..255] float32
    # EfficientNetB3 in Keras has its own internal preprocessing layers.
    base_model = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
    base_model.trainable = (not freeze_encoder)

    # Skip connections
    skip_connection_names = [
        "block2a_expand_activation",
        "block3a_expand_activation",
        "block4a_expand_activation",
        "block6a_expand_activation",
    ]
    encoder_outputs = [base_model.get_layer(name).output for name in skip_connection_names]
    encoder_output = base_model.output

    decoder_filters = [192, 128, 64, 32, 16]

    x = encoder_output
    for i in range(4):
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, encoder_outputs[3 - i]])
        x = residual_block(x, decoder_filters[i], dropout_rate=dropout_rate)
        x = cbam_module(x)

    x = UpSampling2D((2, 2))(x)
    x = residual_block(x, decoder_filters[4], dropout_rate=dropout_rate)
    x = cbam_module(x)

    outputs = Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(x)
    return Model(inputs, outputs, name="BetterNet")
