import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Activation, BatchNormalization, UpSampling2D,
                                     Input, Concatenate, Add, Dropout, GlobalAveragePooling2D,
                                     Reshape, Dense, multiply, GlobalMaxPooling2D, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.regularizers import l1_l2


def channel_attention_module(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal',
                             use_bias=True, bias_initializer='zeros')
    shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    attention_feature = Add()([avg_pool, max_pool])
    attention_feature = Activation('sigmoid')(attention_feature)
    return multiply([input_feature, attention_feature])


def spatial_attention_module(input_feature):
    kernel_size = 7
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention_feature = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same',
                              activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    return multiply([input_feature, attention_feature])


def cbam_module(input_feature, ratio=8):
    x = channel_attention_module(input_feature, ratio)
    x = spatial_attention_module(x)
    return x


def squeeze_excitation_module(input_feature, ratio=16):
    filters = input_feature.shape[-1]
    se_shape = (1, 1, filters)

    squeeze = GlobalAveragePooling2D()(input_feature)
    squeeze = Reshape(se_shape)(squeeze)
    excitation = Dense(filters // ratio, activation='relu')(squeeze)
    excitation = Dense(filters, activation='sigmoid')(excitation)
    return multiply([input_feature, excitation])


def residual_block(input_feature, num_filters, dropout_rate=0.5):
    x = input_feature
    x = Conv2D(num_filters // 4, (1, 1), padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters // 4, (3, 3), padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(num_filters, (1, 1), padding="same", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input_feature)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    x = squeeze_excitation_module(x)
    x = Dropout(dropout_rate)(x)
    return x


def BetterNet(input_shape=(224, 224, 3), num_classes=1, dropout_rate=0.5, freeze_encoder=True):
    inputs = Input(input_shape)

    # Build EfficientNet-B3
    # NOTE: We feed [0,1] images (data.py), so we DO NOT rely on internal preprocessing assumptions.
    base_model = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
    base_model.trainable = (not freeze_encoder)

    # Skip connections (same as your repo)
    skip_connection_names = [
        "block2a_expand_activation",
        "block3a_expand_activation",
        "block4a_expand_activation",
        "block6a_expand_activation"
    ]
    encoder_outputs = [base_model.get_layer(name).output for name in skip_connection_names]
    encoder_output = base_model.output

    # Decoder filters (KEEP as repo)
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
