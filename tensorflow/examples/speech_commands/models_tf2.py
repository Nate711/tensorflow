import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential
import matplotlib.pyplot as plt


def create_model(
    model_settings,
    model_architecture,
):
    if model_architecture == "conv":
        return create_conv(model_settings)
    elif model_architecture == "tiny_conv":
        return create_tiny_conv(model_settings)
    else:
        return None


def create_tiny_conv(model_settings):
    input_frequency_size = model_settings["fingerprint_width"]
    input_time_size = model_settings["spectrogram_length"]
    dropout_rate = 0.2
    label_count = model_settings["label_count"]
    conv_stride = (2, 2)
    conv_filter_size = (8, 10)
    conv_num_filters = 8

    model = Sequential(
        [
            layers.Reshape(
                [input_time_size, input_frequency_size, 1],
                input_shape=[input_time_size * input_frequency_size],
            ),
            layers.DepthwiseConv2D(
                kernel_size=conv_filter_size,
                strides=conv_stride,
                depth_multiplier=conv_num_filters,
                activation="relu",
            ),
            layers.Dropout(dropout_rate),
            layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dense(label_count),
        ]
    )
    return model

def create_conv(model_settings):
    input_frequency_size = model_settings["fingerprint_width"]
    input_time_size = model_settings["spectrogram_length"]
    dropout_rate = 0.2
    label_count = model_settings["label_count"]
    
    ## Conv 1
    conv_stride_layer1 = (1, 1)
    conv_filter_size_layer1 = (8, 20)
    conv_num_filters_layer1 = 8

    ## Conv2
    conv_stride_layer2 = (1, 1)
    conv_filter_size_layer2 = (4, 10)
    conv_num_filters_layer2 = 8

    model = Sequential(
        [
            layers.Reshape(
                [input_time_size, input_frequency_size, 1],
                input_shape=[input_time_size * input_frequency_size],
            ),
            layers.DepthwiseConv2D(
                kernel_size=conv_filter_size_layer1,
                strides=conv_stride_layer1,
                depth_multiplier=conv_num_filters_layer1,
                activation="relu",
                padding="valid",
            ),
            layers.Dropout(dropout_rate),
            layers.MaxPool2D(),
            layers.DepthwiseConv2D(
                kernel_size=conv_filter_size_layer2,
                strides=conv_stride_layer2,
                depth_multiplier=conv_num_filters_layer2,
                activation="relu",
                padding="valid",
            ),
            layers.Dropout(dropout_rate),
            # layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dense(label_count),
        ]
    )
    return model
