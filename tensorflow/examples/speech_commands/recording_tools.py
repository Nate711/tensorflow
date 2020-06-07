from attributedict.collections import AttributeDict
import math

# Audio
import pyaudio
import wave
from sys import byteorder
from array import array
from struct import pack

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential

# Custom
import models_tf2

# You can tune these
THRESHOLD = 1000  # silence threshold, on -32k to 32k scale
MAXIMUM = 2 ** 15 * 0.75  # volume scaling

# Don't mess
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
INT16_MULTIPLIER = 2 ** 15 - 1.0
RATE = 16000
WINDOW_FRAMES = int(RATE * 0.030)
STEP_FRAMES = int(RATE * 0.010)
MFCCS_TO_KEEP = 40

# for original model
LABEL_COUNT = 12


def add_beginning_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array("h", silence)
    r.extend(snd_data)
    return r


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


def record(duration=1.0):
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

            if chunks >= duration * RATE / CHUNK_SIZE:
                break

    print("Done.", flush=True)

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = add_beginning_silence(r, 0.2)
    r = normalize(r)
    r = r[0 : int(duration * RATE)]
    return sample_width, r


def normalized_audio_to_mfcc(audio):
    stfts = tf.signal.stft(audio, frame_length=WINDOW_FRAMES, frame_step=STEP_FRAMES)
    spectrogram = tf.square(tf.abs(stfts))
    # Also works well without squaring...
    # spectrogram = tf.abs(stfts)

    # get mel-scale
    num_spectrogram_bins = spectrogram.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = (
        20.0,
        4000.0,
        MFCCS_TO_KEEP,
    )  # Set to the defaults for tf.ops.mfcc
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

    # For some reason, does not give the same result as the mfccs. Average euclidean distance is 550
    # When using mfcc2, the predictor also behaves very poorly
    # making the magnitude squared doesn't seem to affect much...
    # spec2 = tf.raw_ops.AudioSpectrogram(input=audio[:, tf.newaxis], window_size=WINDOW_FRAMES, stride=STEP_FRAMES, magnitude_squared=False)
    # mfcc2 = tf.raw_ops.Mfcc(spectrogram=spec2, sample_rate=RATE, dct_coefficient_count=40)
    # print("norm diff: ",end="")
    # print(tf.norm(mfcc2-mfccs))

    return mfccs


def normalize_audio(data):
    raw = np.frombuffer(data, dtype=np.int16).astype(float) / INT16_MULTIPLIER
    audio = tf.convert_to_tensor(raw, dtype=tf.float32)
    return audio


def save_mfcc(numpy_path, wav_path):
    "Records from the microphone and outputs the resulting data to 'path'"

    # Record live audio from microphone
    sample_width, data = record()
    # Convert from int16 to [-1,1] float
    audio = normalize_audio(data)
    # get mffcs from raw audio
    mfccs = normalized_audio_to_mfcc(audio)

    np.save(numpy_path, mfccs, allow_pickle=False)

    data = pack("<" + ("h" * len(data)), *data)
    wf = wave.open(wav_path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

    return mfccs


if __name__ == "__main__":
    model_settings = {
        "fingerprint_width": 40,
        "spectrogram_length": 98,
        "label_count": 12,
    }
    model = models_tf2.create_conv(model_settings)
    embedding_model = Sequential(model.layers[:-1])

    checkpoint_path = "training/20200604-213003/cp-75"
    model.load_weights(checkpoint_path)

    folder = "random"
    start_i = 20
    for i in range(start_i, 20 + start_i):
        out = save_mfcc(
            f"recordings/{folder}/mfcc{i}.npy", f"recordings/{folder}/recording_{i}.wav"
        )

        # Kind of dumb given that the model will just reshape to 98x40 anyways
        input_tensor = tf.reshape(out, [1, 3920])

        embedding = embedding_model(input_tensor)
        np.save(f"recordings/{folder}/embedding{i}.npy", embedding, allow_pickle=False)

        label_idx = np.argmax(model(input_tensor), axis=1)[0]
        # print(f"label index: {label_idx}")
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
