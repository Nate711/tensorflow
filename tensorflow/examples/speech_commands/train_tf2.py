import tensorflow as tf
import argparse
import os.path
import sys

import models
import models_tf2
import input_data

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_url",
        type=str,
        # pylint: disable=line-too-long
        default="https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        # pylint: enable=line-too-long
        help="Location of speech training data archive on the web.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tmp/speech_dataset/",
        help="""\
    Where to download the speech training data to.
    """,
    )
    parser.add_argument(
        "--background_volume",
        type=float,
        default=0.1,
        help="""\
    How loud the background noise should be, between 0 and 1.
    """,
    )
    parser.add_argument(
        "--background_frequency",
        type=float,
        default=0.8,
        help="""\
    How many of the training samples have background noise mixed in.
    """,
    )
    parser.add_argument(
        "--silence_percentage",
        type=float,
        default=10.0,
        help="""\
    How much of the training data should be silence.
    """,
    )
    parser.add_argument(
        "--unknown_percentage",
        type=float,
        default=10.0,
        help="""\
    How much of the training data should be unknown words.
    """,
    )
    parser.add_argument(
        "--time_shift_ms",
        type=float,
        default=100.0,
        help="""\
    Range to randomly shift the training audio by in time.
    """,
    )
    parser.add_argument(
        "--testing_percentage",
        type=int,
        default=10,
        help="What percentage of wavs to use as a test set.",
    )
    parser.add_argument(
        "--validation_percentage",
        type=int,
        default=10,
        help="What percentage of wavs to use as a validation set.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Expected sample rate of the wavs",
    )
    parser.add_argument(
        "--clip_duration_ms",
        type=int,
        default=1000,
        help="Expected duration in milliseconds of the wavs",
    )
    parser.add_argument(
        "--window_size_ms",
        type=float,
        default=30.0,
        help="How long each spectrogram timeslice is.",
    )
    parser.add_argument(
        "--window_stride_ms",
        type=float,
        default=10.0,
        help="How far to move in time between spectrogram timeslices.",
    )
    parser.add_argument(
        "--feature_bin_count",
        type=int,
        default=40,
        help="How many bins to use for the MFCC fingerprint",
    )
    parser.add_argument(
        "--how_many_training_steps",
        type=str,
        default="15000,3000",
        help="How many training loops to run",
    )
    parser.add_argument(
        "--eval_step_interval",
        type=int,
        default=400,
        help="How often to evaluate the training results.",
    )
    parser.add_argument(
        "--learning_rate",
        type=str,
        default="0.001,0.0001",
        help="How large a learning rate to use when training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="How many items to train with at once",
    )
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default="/tmp/retrain_logs",
        help="Where to save summary logs for TensorBoard.",
    )
    parser.add_argument(
        "--wanted_words",
        type=str,
        default="yes,no,up,down,left,right,on,off,stop,go",
        help="Words to use (others will be added to an unknown label)",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/tmp/speech_commands_train",
        help="Directory to write event logs and checkpoint.",
    )
    parser.add_argument(
        "--save_step_interval",
        type=int,
        default=100,
        help="Save model checkpoint every save_steps.",
    )
    parser.add_argument(
        "--start_checkpoint",
        type=str,
        default="",
        help="If specified, restore this pretrained model before any training.",
    )
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="conv",
        help="What model architecture to use",
    )
    parser.add_argument(
        "--check_nans",
        type=bool,
        default=False,
        help="Whether to check for invalid numbers during processing",
    )
    parser.add_argument(
        "--quantize",
        type=bool,
        default=False,
        help="Whether to train the model for eight-bit deployment",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="mfcc",
        help='Spectrogram processing mode. Can be "mfcc", "average", or "micro"',
    )

    # Function used to parse --verbosity argument
    def verbosity_arg(value):
        """Parses verbosity argument.

        Args:
        value: A member of tf.logging.
        Raises:
        ArgumentTypeError: Not an expected value.
        """
        value = value.upper()
        if value == "DEBUG":
            return tf.compat.v1.logging.DEBUG
        elif value == "INFO":
            return tf.compat.v1.logging.INFO
        elif value == "WARN":
            return tf.compat.v1.logging.WARN
        elif value == "ERROR":
            return tf.compat.v1.logging.ERROR
        elif value == "FATAL":
            return tf.compat.v1.logging.FATAL
        else:
            raise argparse.ArgumentTypeError("Not an expected value")

    parser.add_argument(
        "--verbosity",
        type=verbosity_arg,
        default=tf.compat.v1.logging.INFO,
        help='Log verbosity. Can be "DEBUG", "INFO", "WARN", "ERROR", or "FATAL"',
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="gradient_descent",
        help="Optimizer (gradient_descent or momentum)",
    )

    FLAGS, unparsed = parser.parse_known_args()

    # Set the verbosity based on flags (default is INFO, so we see all messages)
    tf.compat.v1.logging.set_verbosity(FLAGS.verbosity)

    # Start a new TensorFlow session.
    sess = tf.compat.v1.InteractiveSession()

    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(","))),
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.feature_bin_count,
        FLAGS.preprocess,
    )
    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url,
        FLAGS.data_dir,
        FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(","),
        FLAGS.validation_percentage,
        FLAGS.testing_percentage,
        model_settings,
        FLAGS.summaries_dir,
    )
    fingerprint_size = model_settings["fingerprint_size"]
    label_count = model_settings["label_count"]
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

    model = models_tf2.create_model(
        None,
        model_settings,
        model_architecture="tiny_conv",
        is_training=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # List of metrics to monitor
        metrics=["sparse_categorical_accuracy"],
    )

    steps = 1000
    for step in range(steps):
        train_fingerprints, train_ground_truth = audio_processor.get_data(
            how_many=FLAGS.batch_size,
            offset=0,
            model_settings=model_settings,
            background_frequency=FLAGS.background_frequency,
            background_volume_range=FLAGS.background_volume,
            time_shift=time_shift_samples,
            mode="training",
            sess=sess,
        )
        model.train_on_batch(train_fingerprints, train_ground_truth)
        print(step, ' ', end='')
