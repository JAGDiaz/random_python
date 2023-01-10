import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
from models import streetsigns_model_creator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=.1):
    
    folders = os.listdir(path_to_data)
    for folder in folders:
        full_path = os.path.join(path_to_data, folder)
        images_paths = glob.glob(os.path.join(full_path, "*.png"))

        x_train, x_val = train_test_split(images_paths, test_size=split_size)

        for x in x_train:

            path_to_folder = os.path.join(path_to_save_train, folder)
            if not os.path.isdir(path_to_folder): os.makedirs(path_to_folder)
            shutil.copy(x, path_to_folder)

        for x in x_val:
            path_to_folder = os.path.join(path_to_save_val, folder)
            if not os.path.isdir(path_to_folder): os.makedirs(path_to_folder)
            shutil.copy(x, path_to_folder)

def order_test_set(path_to_images, path_to_csv):

    try:
        with open(path_to_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
    
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                imag_name = row[-1].replace('Test/', '')
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)
                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, imag_name)
                shutil.move(img_full_path, path_to_folder)
    
    except:
        print('[INFO] : Error reading csv file.')
    
    finally:
        pass

def create_image_data_generators(batch_size, path_to_train_data, path_to_val_data, path_to_test_data):

    preprocessor = ImageDataGenerator(
        rescale=1/255., rotation_range=10,
        width_shift_range=0.1, height_shift_range=0.1
    )

    train_data_generator = preprocessor.flow_from_directory(
        path_to_train_data, class_mode="categorical", target_size=(60,60),
        color_mode='rgb', shuffle=True, batch_size=batch_size
    )

    val_data_generator = preprocessor.flow_from_directory(
        path_to_val_data, class_mode="categorical", target_size=(60,60),
        color_mode='rgb', shuffle=False, batch_size=batch_size
    )

    test_data_generator = preprocessor.flow_from_directory(
        path_to_test_data, class_mode="categorical", target_size=(60,60),
        color_mode='rgb', shuffle=False, batch_size=batch_size
    )

    return train_data_generator, val_data_generator, test_data_generator

if __name__ == "__main__":

    here = os.getcwd()

    path_to_train = os.path.join(here, "german_signage", "Training_data", "train")
    path_to_val = os.path.join(here, "german_signage", "Training_data", "val")
    path_to_test = os.path.join(here, "german_signage", "Test")
    batch_size = 128; epochs = 15

    train_data_generator, val_data_generator, test_data_generator = \
        create_image_data_generators(batch_size, path_to_train, path_to_val, path_to_test)
    number_of_classes = train_data_generator.num_classes

    training = False
    testing = True

    if training:
        path_to_save_model = os.path.join(here, "Models")

        best_accuracy_checkpoint_saver = ModelCheckpoint(
            path_to_save_model, monitor='val_accuracy', mode='max', save_best_only=True,
            save_freq='epoch', verbose=1
        )

        stop_early_callback = EarlyStopping(
            monitor='val_accuracy', patience=10
        )

        model = streetsigns_model_creator(number_of_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(
            train_data_generator, epochs=epochs, batch_size=batch_size, 
            validation_data=val_data_generator, 
            callbacks=[best_accuracy_checkpoint_saver, stop_early_callback]
        )

    if testing:
        model = load_model(os.path.join(here, "Models"))
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_data_generator)

        print("Evaluating testing set:")
        model.evaluate(test_data_generator)