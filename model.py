import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB3

def cbam_module(input_feature, ratio=8):
    """
    Convolutional Block Attention Module (CBAM) - Original Paper Implementation
    """
    channel = input_feature.shape[-1]
    
    # Channel Attention
    s1 = layers.Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    s2 = layers.Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    
    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = s2(s1(avg_pool))
    
    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = s2(s1(max_pool))
    
    channel_attention = layers.Activation('sigmoid')(layers.Add()([avg_pool, max_pool]))
    x = layers.Multiply()([input_feature, channel_attention])
    
    # Spatial Attention
    avg_pool = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    
    spatial_attention = layers.Conv2D(1, 7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    
    return layers.Multiply()([x, spatial_attention])

def residual_block(input_tensor, filters, strides=1):
    x = layers.Conv2D(filters, 3, strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, 3, strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    if strides != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = input_tensor
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def decoder_block(input_tensor, skip_tensor, filters, dropout_rate=0.0):
    x = layers.UpSampling2D((2, 2))(input_tensor)
    
    if skip_tensor is not None:
        if x.shape[1] != skip_tensor.shape[1] or x.shape[2] != skip_tensor.shape[2]:
            x = layers.Resizing(skip_tensor.shape[1], skip_tensor.shape[2])(x)
        x = layers.Concatenate()([x, skip_tensor])
    
    x = residual_block(x, filters)
    x = residual_block(x, filters)
    x = cbam_module(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
        
    return x

def BetterNet(input_shape=(256, 256, 3), num_classes=1, dropout_rate=0.5):
    inputs = Input(shape=input_shape)
    
    # Original Paper uses EfficientNetB3
    encoder = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Encoder Output Features
    skip1 = encoder.get_layer('block6a_expand_activation').output 
    skip2 = encoder.get_layer('block4a_expand_activation').output 
    skip3 = encoder.get_layer('block3a_expand_activation').output 
    skip4 = encoder.get_layer('block2a_expand_activation').output 
    
    # Decoder Path
    d1 = decoder_block(encoder.output, skip1, 256, dropout_rate)
    d2 = decoder_block(d1, skip2, 128, dropout_rate)
    d3 = decoder_block(d2, skip3, 64, dropout_rate)
    d4 = decoder_block(d3, skip4, 32, dropout_rate)
    d5 = decoder_block(d4, None, 16, dropout_rate)
    
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(d5)
    
    return models.Model(inputs=inputs, outputs=outputs, name="BetterNet")
