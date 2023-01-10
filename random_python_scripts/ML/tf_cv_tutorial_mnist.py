import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from models import seq_model, custom_model, functional_model
from utils import sparse_to_non, act_func_plotter, display_some_examples

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    new_y_train = tf.keras.utils.to_categorical(y_train, 10)
    new_y_test = tf.keras.utils.to_categorical(y_test, 10)

    print(new_y_train)

    print(f"{x_train.shape = }")
    print(f"{y_train.shape = }")
    print(f"{new_y_train.shape = }")
    print(f"{x_test.shape = }")
    print(f"{y_test.shape = }")

    # display_some_examples(x_train, y_train)

    # Normalize unsigned integer values to [0,1] floats
    x_train = x_train.astype('float32') / 255.; x_test = x_test.astype('float32') / 255.
    x_train = x_train[...,np.newaxis]; x_test = x_test[...,np.newaxis]

    my_model = custom_model()
    my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    # Training on train set
    train = my_model.fit(x=x_train, y=new_y_train, batch_size=100, epochs=3, validation_split=.2)

    # Evaluation on test set
    test = my_model.evaluate(x=x_test, y=new_y_test, batch_size=100)

    train_dict = train.history

    