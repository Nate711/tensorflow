import recording_tools
import embedding_analysis
import models_tf2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential

import UDPComms

def get_samples(N=30):
    mfccs = np.zeros((N, 98, 40))
    for i in range(N):
        print("Sample: ", i, end=". ")
        sample_width, data = recording_tools.record(1.0)
        data = recording_tools.normalize_audio(data)
        mfcc = recording_tools.normalized_audio_to_mfcc(data)
        mfccs[i, :, :] = mfcc
    return mfccs


def save_mfccs(mfccs):
    np.save("recordings/saves/mfccs.npy", mfccs)


def get_transfer_model():
    # didn't want to go through the whole settings process, but not as portable
    model_settings = {
        "fingerprint_width": 40,
        "spectrogram_length": 98,
        "label_count": 12,
    }

    # Make the base out of the pre-trained command network
    # minus the classification layer and make the weights
    # non-trainable
    base = models_tf2.create_conv(model_settings)
    checkpoint_path = "training/20200604-213003/cp-75"
    base.load_weights(checkpoint_path)
    embedding_model = Sequential(base.layers[:-1])
    embedding_model.trainable = False

    # Stack on a new name classifier
    name_classifier = embedding_analysis.get_name_classifier()
    transfer_model = Sequential(embedding_model.layers + name_classifier.layers)
    transfer_model.summary()

    return transfer_model


def train_transfer_model(model, X, y, X_val, y_val):
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )
    model.fit(X, y, epochs=20, batch_size=32)

    print("\nEVALUATION")
    model.evaluate(x=X_val, y=y_val)

    out = model.predict(X_val)
    out = np.argmax(out, axis=1)

    true_pos = np.sum((out == 1) & (y_val == 1))  # / np.sum(y_val == 1)
    n_pos = np.sum(y_val == 1)
    true_neg = np.sum((out == 0) & (y_val == 0))  # / np.sum(y_val == 0)
    n_negs = np.sum(y_val == 0)

    print("\n\n**********************************\n\nEVALUATION RESULTS")
    print(f"True positive rate: {true_pos}/{n_pos}")
    print(f"True negative rate: {true_neg}/{n_negs}")


def load_mfcc(folder, n):
    # loads mfccs from folder
    # returns numpy array num_samples x num_features
    # num_features = 3920 (ie flattened)

    data = np.zeros((n, 3920))
    for i in range(n):
        data[i, :] = np.load(f"{folder}/mfcc{i}.npy").flatten()
    return data


def load_collected_mfcc(path):
    return np.load(path)


def train_on_name(model, name, repeats=2, speechcommand_samples=100):
    # create data points for "name" utterances
    name = np.tile(name, (repeats, 1))
    n_name = name.shape[0]

    # create background noise utterances
    random = load_mfcc("recordings/random", 60)
    noise = load_collected_mfcc("recordings/speech_commands/X.npy")[
        0:speechcommand_samples, :
    ]
    noise = np.vstack((noise, random))
    n_noise = noise.shape[0]

    # construct data matrices
    X = np.vstack((name, noise))
    y = np.concatenate((np.ones(n_name), np.zeros(n_noise)), axis=0)

    # construct validation matrices
    name_test = load_mfcc("recordings/pupper_test", 20)
    other = load_mfcc("recordings/other", 20)
    go = load_mfcc("recordings/go", 20)
    X_val = np.vstack((name_test, other, go))
    y_val = np.concatenate(
        (
            np.ones(name_test.shape[0]),
            np.zeros(other.shape[0]),
            np.zeros(go.shape[0]),
        ),
        axis=0,
    )

    # train the model
    train_transfer_model(model, X, y, X_val, y_val)


def train_on_live(N=20):
    model = get_transfer_model()
    mfccs = get_samples(N)
    mfccs = tf.reshape(mfccs, [mfccs.shape[0], 3920])
    train_on_name(model, mfccs, repeats=3, speechcommand_samples=300)

    # save it to file
    model.save("transfer_model/fit")

def fake_train_on_live():
    model = get_transfer_model()
    mfccs = load_mfcc("recordings/pupper", 20)
    train_on_name(model, mfccs, repeats=3, speechcommand_samples=60)

    model.save("transfer_model/fit_recorded")

def train_on_recorded():
    model = get_transfer_model()
    mfccs = load_mfcc("recordings/pupper", 60)
    train_on_name(model, mfccs, repeats=5, speechcommand_samples=1200)

    model.save("transfer_model/fit_recorded")


def test_model_live(path):
    pub = UDPComms.Publisher(8008)

    model = tf.keras.models.load_model(path)
    while 1:
        mfcc = get_samples(1)
        mfcc = tf.reshape(mfcc, [mfcc.shape[0], 3920])
        logits = model.predict(mfcc)
        label_idx = np.argmax(logits, axis=1)[0]
        labels=["noise", "name"]
        print("HEARD: ", end="")
        heard = labels[label_idx]
        print(heard)

        pub.send(heard)

if __name__ == "__main__":
    train_on_live(N=20)
    test_model_live("transfer_model/fit")

    # fake_train_on_live()

    # train_on_recorded()
    # test_model_live("transfer_model/fit_recorded")
