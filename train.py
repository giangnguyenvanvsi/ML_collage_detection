# import tensorflow as tf
import keras
from keras import layers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
from srm import srm_filter
import argparse

from keras.callbacks import ModelCheckpoint


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='./data' ,required=False, help="Path to the training data")
args = vars(ap.parse_args())


path_data_train = os.path.join(args['dataset'],'train')
path_data_validation = os.path.join(args['dataset'],'valid')
path_data_test = os.path.join(args['dataset'],'test')
path_checkpoints = './checkpoints'

if not os.path.exists(path_data_train):
    os.makedirs(path_data_train)
if not os.path.exists(path_data_validation):
    os.makedirs(path_data_validation)
if not os.path.exists(path_data_test):
    os.makedirs(path_data_test)
if not os.path.exists(path_checkpoints):
    os.makedirs(path_checkpoints)

train_batches = ImageDataGenerator().flow_from_directory(path_data_train, target_size=(224,224), classes=['single', 'collage'], batch_size=16, shuffle=True)
test_batches =  ImageDataGenerator().flow_from_directory(path_data_validation , target_size=(224,224), classes=['single', 'collage'], batch_size=4)
valid_batches = ImageDataGenerator().flow_from_directory(path_data_test, target_size=(224,224), classes=['single', 'collage'], batch_size=8, shuffle=True)


''' Image Augmentation '''
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         preprocessing_function=srm_filter)
# val_datagen = ImageDataGenerator(rescale=1./255, 
#                                 preprocessing_function=srm_filter)

# train_generator = train_datagen.flow_from_directory(
#         path_data_train,
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='binary')
# validation_generator = val_datagen.flow_from_directory(
#         path_data_validation,
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='binary')

if __name__ == '__main__':
    
    input_shape = (224, 224, 3)
    # reload pre-trained model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(Adam(lr=.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    filepath = os.path.join(path_checkpoints, "weights-improvement-epoch{epoch:02d}-{val_accuracy:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    ''' Fit the model with preprocess image in Keras '''
    # model.fit(train_generator,
    #         steps_per_epoch=2000,
    #         epochs=50,
    #         validation_data=validation_generator,
    #         validation_steps=800, 
    #         batch_size=10,
    #         verbose=0, 
    #         callbacks=callbacks_list)

    model.fit_generator(train_batches, 
                        steps_per_epoch=14, 
                        validation_data=valid_batches, 
                        validation_steps=10, 
                        epochs=20, 
                        verbose=2, 
                        callbacks=callbacks_list)











