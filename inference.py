import numpy as np
import cv2
import keras
from keras import layers
import argparse
from srm import srm_filter

input_shape = (224, 224, 3)
id2label = {0: "NON-COLLAGE", 1: "COLLAGE"}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-c", "--checkpoint", default= 'pretrained-model/CollageDetectionModelWeights.h5',required=False, help="Path to the image")

    args = vars(ap.parse_args())
    weights_path = args['checkpoint']
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

    model.load_weights(weights_path)
    path_image = args["image"]
    image_ = cv2.imread(path_image)
    # inferencing
    image = srm_filter(image_)
    predicted_id = np.argmax(model.predict(image)[0])
    print(f"Predicted label: {id2label[predicted_id]} with trust {np.max(model.predict(image)[0])}")