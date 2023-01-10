import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, \
                                    Flatten, GlobalAvgPool2D

# Tensorflow.keras.Sequential
def seq_model():
    model = tf.keras.Sequential(
        [
            Input(shape=(28,28,1)),
            Conv2D(32, (3,3), activation='relu'),
            Conv2D(64, (3,3), activation='relu'),
            MaxPool2D(),
            BatchNormalization(), 

            Conv2D(128, (3,3), activation='relu'),
            MaxPool2D(),
            BatchNormalization(), 

            GlobalAvgPool2D(),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ]

    )
    return model


# Tensorflow.keras.Functional: Function that returns a model.
def functional_model():

    my_input = Input(shape=(28,28,1))
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    return tf.keras.Model(inputs=my_input, outputs=x)


# Tensorflow.keras.Model: Inherit from the Model class.
class custom_model(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()

        # Input(shape=(28,28,1))
        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()

        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()

        self.globalaveragepool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)

        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)

        x = self.globalaveragepool1(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
        

# Functional approach for german signage
def streetsigns_model_creator(number_of_sign_classes, image_width_pixels=60, 
                              image_height_pixels=60, number_of_channels=3):

    my_input_layer = Input(shape=(image_width_pixels, image_height_pixels, number_of_channels))

    x = Conv2D(32, (3,3), activation='relu')(my_input_layer)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    # x = Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(number_of_sign_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=my_input_layer, outputs=x)


if __name__ == '__main__':
    
    model = streetsigns_model_creator(10)

    model.summary()