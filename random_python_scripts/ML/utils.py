import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def sparse_to_non(sparse_encoding, number_of_classes=10):
    non_sparse_encoding = np.zeros((sparse_encoding.size, number_of_classes), dtype=np.uint)
    for (row, index) in zip(non_sparse_encoding, sparse_encoding):
        row[index] = 1
    return non_sparse_encoding

def display_some_examples(examples, labels):
    
    fig, ax = plt.subplots(5,5, figsize=(10,10))
    ax = ax.flatten()

    for ii in range(ax.size):
        idx = np.random.randint(0, examples.shape[0])
        img = examples[idx]
        label = labels[idx]

        ax[ii].imshow(img, cmap='gray')
        ax[ii].set_title(f"{label}")

    fig.tight_layout(pad=1)
    plt.show()

def act_func_plotter():

    act_funcs = [i for i in dir(tf.keras.activations) if "__" not in i and i not in ("get", "_sys", "deserialize", "serialize")]

    x = tf.linspace(-5,5, 10001)

    for identifier in act_funcs:
        try:
            func = tf.keras.activations.get(identifier)
            y = func(x)
            plt.plot(x,y)
            plt.title(identifier)
            plt.show()
        except:
            print(f"{identifier} done fucked up!")
