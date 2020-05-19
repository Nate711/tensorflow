import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential
import matplotlib.pyplot as plt


def create_model(
    fingerprint_input,
    model_settings,
    model_architecture,
    is_training,
    runtime_settings=None,
):
    return create_tiny_conv(
        fingerprint_input, model_settings, is_training, runtime_settings
    )


def create_tiny_conv(fingerprint_input, model_settings, is_training, runtime_settings):
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
            layers.Flatten(),
            layers.Dense(label_count),
        ]
    )
    return model
