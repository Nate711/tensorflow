from sys import byteorder
from array import array
from struct import pack

from attributedict.collections import AttributeDict

import pyaudio
import wave
import math

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

import argparse

import models
import models_tf2

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
DESIRED_DURATION = 1.0  # s
MAXIMUM = 2 ** 15 * 0.75
INT16_MULTIPLIER = 2 ** 15 - 1.0


WINDOW_FRAMES = int(RATE * 0.030)
STEP_FRAMES = int(RATE * 0.010)


def prepare_model_settings(
    label_count,
    sample_rate,
    clip_duration_ms,
    window_size_ms,
    window_stride_ms,
    feature_bin_count,
    preprocess,
):
    """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    feature_bin_count: Number of frequency bins to use for analysis.
    preprocess: How the spectrogram is processed to produce features.

  Returns:
    Dictionary containing common settings.

  Raises:
    ValueError: If the preprocessing mode isn't recognized.
  """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    if preprocess == "average":
        fft_bin_count = 1 + (_next_power_of_two(window_size_samples) / 2)
        average_window_width = int(math.floor(fft_bin_count / feature_bin_count))
        fingerprint_width = int(math.ceil(fft_bin_count / average_window_width))
    elif preprocess == "mfcc":
        average_window_width = -1
        fingerprint_width = feature_bin_count
    elif preprocess == "micro":
        average_window_width = -1
        fingerprint_width = feature_bin_count
    else:
        raise ValueError(
            'Unknown preprocess mode "%s" (should be "mfcc",'
            ' "average", or "micro")' % (preprocess)
        )
    fingerprint_size = fingerprint_width * spectrogram_length
    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "fingerprint_width": fingerprint_width,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": sample_rate,
        "preprocess": preprocess,
        "average_window_width": average_window_width,
    }


def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    "Average the volume out"
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array("h")
    for i in snd_data:
        r.append(int(i * times))
    return r


def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    num_silent = 0
    snd_started = False

    r = array("h")

    chunks = 0
    print("Waiting for voice...", end="", flush=True)
    while 1:
        # little endian, signed short
        snd_data = array("h", stream.read(CHUNK_SIZE))
        if byteorder == "big":
            snd_data.byteswap()

        if not is_silent(snd_data) and not snd_started:
            print("Recording...", end="", flush=True)
            snd_started = True

        if snd_started:
            r.extend(snd_data)
            chunks += 1

            if chunks >= DESIRED_DURATION * RATE / CHUNK_SIZE:
                break

    print("Done.", flush=True)

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = r[
        0 : int(DESIRED_DURATION * RATE)
    ]  # get rid of the bytes at the end due to chunk size
    return sample_width, r


def save_mfcc_tensor(path):
    "Records from the microphone and outputs the resulting data to 'path'"

    # Record live audio from microphone
    sample_width, data = record()

    # convert from short array to float and scale to [-1,  1]
    raw = np.frombuffer(data, dtype=np.int16).astype(float) / INT16_MULTIPLIER
    audio = tf.convert_to_tensor(raw, dtype=tf.float32)

    # get spectrogram
    stfts = tf.signal.stft(audio, frame_length=WINDOW_FRAMES, frame_step=STEP_FRAMES)
    spectrogram = tf.abs(stfts)

    # get mel-scale
    num_spectrogram_bins = spectrogram.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 20.0, 4000.0, 40
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, RATE, lower_edge_hertz, upper_edge_hertz,
    )
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
    )

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)

    np.save(path, mfccs, allow_pickle=False)

    return mfccs


if __name__ == "__main__":
    FLAGS = AttributeDict(
        {
            "sample_rate": 1600,
            "clip_duration_ms": 1000,
            "window_size_ms": 30,
            "window_stride_ms": 10,
            "feature_bin_count": 40,
            "preprocess": "mfcc",
            "model_architecture": "conv",
            "wanted_words": "yes,no,up,down,left,right,on,off,stop,go",
        }
    )

    model_settings = prepare_model_settings(
        12,
        FLAGS.sample_rate,
        FLAGS.clip_duration_ms,
        FLAGS.window_size_ms,
        FLAGS.window_stride_ms,
        FLAGS.feature_bin_count,
        FLAGS.preprocess,
    )

    model = models_tf2.create_model(model_settings, FLAGS.model_architecture)

    checkpoint_path = "training/20200604-213003/cp-75"
    model.load_weights(checkpoint_path)

    for i in range(10):
        out = save_mfcc_tensor(f"recordings/mfcc{i}.npy")
        print(out)
        input_tensor = tf.reshape(out, [1, 3920])
        print(input_tensor)
        label_idx = np.argmax(model.predict(input_tensor), axis=1)[0]
        print(f"label index: {label_idx}")
        labels = [
            "silence",
            "unknown",
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
        ]
        print("Heard: ", end="")
        print(labels[label_idx])
