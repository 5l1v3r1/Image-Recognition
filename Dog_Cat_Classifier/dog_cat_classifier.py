import tkMessageBox
import cv2
from pathlib import Path
import tkinter as tk
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
# for gui
from tkinter import filedialog

def train():
        img_width, img_height = 150, 150

        train_data_dir = 'Data/Train' # 1024 images per class
        validation_data_dir = 'Data/Validation' # 416 images per class

        # used to rescale the pixel values from [0, 255] to [0, 1] interval
        datagen = ImageDataGenerator(rescale=1./255)

        # automagically retrieve images and their classes for train and validation sets
        train_generator = datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_width, img_height),
                batch_size=16,
                class_mode='binary')

        validation_generator = datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_width, img_height),
                batch_size=32,
                class_mode='binary')

        model =  Sequential ()
        model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5)) # to prevent overfitting
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop', #gradient descent optimizer
                      metrics=['accuracy'])
        nb_epoch = 20 # steps to train
        nb_train_samples = 2048 # total image to train
        nb_validation_samples = 832 # total image to validate

        model.fit_generator(
                train_generator,
                samples_per_epoch=nb_train_samples,
                nb_epoch=nb_epoch,
                nb_val_samples=nb_validation_samples)

        model_json = model.to_json()
        with open("./model.json", "w") as json_file:
                json_file.write(model_json)

        model.save_weights("./model.h5")
        print("saved model..! ready to go.")


def result(file_path):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("model.h5")
        loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

        img = cv2.imread(file_path)
        img = cv2.resize(img, (150, 150))
        img = img.reshape(1, 150, 150, 3)

        prediction = loaded_model.predict(img)
        print(prediction)
        if prediction[0][0] == 1:
                result = "It is a DOG!"
        else:
                result = "It is a CAT!"

        tkMessageBox.showinfo("RESULT", result)


def input_popup():
    tkMessageBox.showinfo("DOG OR CAT ","Please upload a cat or dog photo with .jpg extension")
    file_path = filedialog.askopenfilename()
    return file_path

root = tk.Tk()
root.withdraw()
input_image = input_popup()
json_file_path = Path("/home/mbt/PycharmProjects/untitled/model.json")
if json_file_path.is_file():
        result(input_image)
else:
        train()
        result(input_image)