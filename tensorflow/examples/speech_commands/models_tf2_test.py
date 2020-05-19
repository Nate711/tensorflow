# from tensorflow.examples.speech_commands import models_tf2 ## only works if using as a package
import tensorflow as tf
import models_tf2
import numpy as np

if __name__ == "__main__":
    model_settings = {
        "fingerprint_width": 40,
        "spectrogram_length": 100,
        "label_count": 10,
    }
    model_architecture = "tiny_conv"
    is_training = True
    runtime_settings = {}

    fingerprint_input = tf.random.uniform(
        shape=[
            1,
            model_settings["spectrogram_length"] * model_settings["fingerprint_width"],
        ]
    )
    print(type(fingerprint_input))

    model = models_tf2.create_model(
        fingerprint_input,
        model_settings,
        model_architecture,
        is_training,
        runtime_settings,
    )

    out = model(fingerprint_input)
    print(type(out))
    print(out.shape)
    print(out)

    numpy_input = np.random.randn(*fingerprint_input.shape)
    print(type(numpy_input))

    model.summary()
