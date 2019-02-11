
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D , Lambda
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications import VGG16
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import load_model
from utils.viz_tools import plot_results

############# VARIABLES ##########################
data_path = "/home/auv/fishdetection/datasets/dogs_vs_cats"
target_size = (224, 224)
batch_size = 4



################################### DATA PROCESSING TOOLS #########################################


def preprocess_image(im):
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    #im = cv2.resize(im, (224, 224)).astype(np.float32)
    im = (im - vgg_mean)
    return im[:, ::-1] # RGB to BGR



def get_batches(directory, target_size=target_size, batch_size=batch_size):
  datagen = ImageDataGenerator()
  return datagen.flow_from_directory(directory=directory,
                                    target_size=target_size,
                                    batch_size=batch_size,
                                    class_mode='categorical')

########################### MODEL TOOLS #####################################################

def build_custom_VGG16():
    # we initialize the model
    model = Sequential()
    # Conv Block 1
    model.add(Lambda(preprocess_image, input_shape=(224, 224, 3), output_shape=(224, 224, 3)))

    # Conv Block 1
    model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # FC layers
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='softmax'))
    return model


############################  Training #######################################

def train_custom_VGG(model, batches, valid_batches, opt= SGD(lr=10e-5) , train_last=False ):
    model.load_weights(os.path.join(data_path, "vgg16_weights_tf_dim_ordering_tf_kernels.h5"))

    x = Dense(batches.num_classes, activation='softmax')(model.layers[-2].output)
    model = Model(model.input, x)
    for layer in model.layers: layer.trainable = False

    if train_last :
        for layer in model.layers[10:]:
            layer.trainable = True

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(batches, steps_per_epoch=batches.samples // batch_size, epochs=2,
                        validation_data=valid_batches, validation_steps=valid_batches.samples // batch_size)
    model.save(os.path.join(data_path, "trained_model.h5"))

def inference(model, test_batches):
    predictions = model.predict_generator(test_batches, steps=test_batches.samples // (batch_size * 2))

#####ERRONOEUS##############

def show_result(predictions, test_batches):

    for i in predictions.shape[0]:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 1, 1)
        plot_results(i, predictions,test_batches)

