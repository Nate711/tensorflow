import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pairwise_distmatrix(data):
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            diff = data[i, :] - data[j, :]
            dist_matrix[i, j] = np.linalg.norm(diff)
    return dist_matrix


def distances_to_point(data, point):
    n = data.shape[0]
    dists = np.zeros(n)
    for i in range(n):
        dists[i] = np.linalg.norm(data[i, :] - point)
    return dists


def load_embeddings(folder, n):
    data = np.zeros((n, 2688))
    for i in range(n):
        data[i, :] = np.load(f"{folder}/embedding{i}.npy").flatten()
    return data


def distance():
    # average based embedding

    other = load_embeddings("recordings/other", 20)
    pupper = load_embeddings("recordings/pupper", 20)
    right = load_embeddings("recordings/right", 20)
    go = load_embeddings("recordings/go", 20)
    non_pupper = np.concatenate((right, go, other), axis=0)
    pupper_test = load_embeddings("recordings/pupper_test/", 20)
    pupper_avg = np.sum(pupper, axis=0) / 20

    # dot product towards the cluster
    # print("Pupper embeddings dotted with pupper embedding mean: ")
    # print(pupper @ pupper_avg)
    # print(np.sum(pupper @ pupper_avg))

    print(
        "Method: Turn the recorded sample into an embedding and then take the dot product of it with the mean embedding for 'pupper' utterances"
    )
    print(
        "If it is more than a learned threshold, label it as a pupper utterance. This is akin to a single hyperplane classifier."
    )

    print(
        "Threshold: min dot product between training pupper embeddings and their mean:"
    )
    threshold = np.amin(pupper @ pupper_avg)
    print(threshold)

    # print("Other embeddings dotted with pupper embedding mean: ")
    # print(non_pupper @ pupper_avg)
    # print(np.sum(non_pupper @ pupper_avg))

    print("False positives: ", end="")
    false_pos = non_pupper @ pupper_avg > threshold
    num = np.sum(false_pos)
    denom = non_pupper.shape[0]
    print(f"{num}/{denom}={num/denom}")

    print("Test set false negatives: ", end="")
    test = pupper_test @ pupper_avg > threshold
    num = np.sum(test)
    denom = pupper_test.shape[0]
    print(f"{denom-num}/{denom}={1-num/denom}")

    # # distance vector
    # print("distances from pupper embeddings to pupper avg")
    # print(distances_to_point(pupper, pupper_avg))

    # print("distances from other embeddings to pupper avg")
    # print(distances_to_point(other, pupper_avg))


def svd():
    other = load_embeddings("recordings/other", 20)
    pupper = load_embeddings("recordings/pupper", 20)
    right = load_embeddings("recordings/right", 20)
    go = load_embeddings("recordings/go", 20)

    pupper_avg = np.sum(pupper, axis=0) / 20
    u, s, vh = np.linalg.svd(right)
    right_top_eig = vh[0, :]

    u, s, vh = np.linalg.svd(pupper)
    pupper_top_eig = vh[0, :]

    data = np.concatenate((pupper, right, go, other), axis=0)
    data = data - np.sum(data, axis=0) / data.shape[0]
    # print(pupperright, pupperright.shape)

    u, s, vh = np.linalg.svd(data)
    print(s)
    print(vh)

    first = vh[0, :]
    second = vh[1, :]
    third = vh[2, :]

    first_projs = data @ first
    second_projs = data @ second
    third_projs = data @ third

    print(first_projs.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(first_projs[0:20], second_projs[0:20], third_projs[0:20], label="pupper")
    ax.scatter(
        first_projs[20:40], second_projs[20:40], third_projs[20:40], label="right"
    )
    ax.scatter(first_projs[40:60], second_projs[40:60], third_projs[40:60], label="go")
    ax.scatter(
        first_projs[60:80],
        second_projs[60:80],
        third_projs[60:80],
        label="random command words",
    )

    plt.title("Speech embeddings projected onto top 2 principal components")
    plt.legend()
    plt.show()


# distance()
# svd()

import tensorflow as tf
from tensorflow.keras import Sequential, layers, regularizers

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def svm():
    other = load_embeddings("recordings/other", 20)
    pupper = load_embeddings("recordings/pupper", 20)
    right = load_embeddings("recordings/right", 20)
    go = load_embeddings("recordings/go", 20)
    X = np.concatenate((pupper, right, go, other), axis=0)
    y = np.zeros(80)
    y[0:20] = 1

    pupper_test = load_embeddings("recordings/pupper_test/", 20)
    val_x = np.concatenate((pupper_test, other))
    val_y = np.zeros(40)
    val_y[0:20] = 1

    clf = make_pipeline(StandardScaler(), SVC(C=1.0, gamma="auto"))
    clf.fit(X, y)

    print("\n\nSVM")
    print("Train accuracy: ")
    print(np.sum(clf.predict(X) == y) / X.shape[0])
    print("Validation accuracy: ")
    print(np.sum(clf.predict(val_x) == val_y) / val_x.shape[0])


def get_label(logits):
    return np.argmax(logits, axis=1)


def get_name_classifier():
    input_size = 2688
    out1 = 10
    out2 = 2
    model = Sequential(
        [
            layers.Dense(out1, kernel_regularizer=regularizers.l2(0.5)),
            layers.ReLU(),
            layers.Dropout(0.5),
            layers.Dense(out2),
        ]
    )
    return model


def fine_tuning():
    model = get_name_classifier()

    pupper = load_embeddings("recordings/pupper", 40)
    other = load_embeddings("recordings/other", 20)
    right = load_embeddings("recordings/right", 20)
    go = load_embeddings("recordings/go", 20)
    data = np.concatenate((pupper, right, go, other), axis=0)
    ground_truth = np.zeros(100)
    ground_truth[0:40] = 1

    pupper_test = load_embeddings("recordings/pupper_test/", 20)
    random = load_embeddings("recordings/random", 20)
    val_x = np.concatenate((pupper_test, random))
    val_y = np.zeros(40)
    val_y[0:20] = 1

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(data, ground_truth, epochs=20)  # , validation_data=(val_x, val_y))#5)

    print("\nEVALUATION")
    model.evaluate(x=val_x, y=val_y)

    print("True positive rate:  ")
    model.evaluate(x=pupper_test, y=np.ones(20))

    print("True negative rate: ")
    model.evaluate(x=random, y=np.zeros(20))


if __name__ == "__main__":
    # distance()
    fine_tuning()
    # svm()
