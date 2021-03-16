import numpy as np
import cv2
import keras
from keras import layers
import argparse

# SRM filters ===========================================
filter1 = np.array([[0, 0, 0, 0, 0],
                    [0, -1, 2, -1, 0],
                    [0, 2, -4, 2, 0],
                    [0, -1, 2, -1, 0],
                    [0, 0, 0, 0, 0]], np.float32)
filter2 = np.array([[-1, 2, -2, 2, -1],
                    [2, -6, 8, -6, 2],
                    [-2, 8, -12, 8, -2],
                    [2, -6, 8, -6, 2],
                    [-1, 2, -2, 2, -1]], np.float32)
filter3 = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, -2, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]], np.float32)
# ========================================================

input_shape = (224, 224, 3)
WEIGHTS_PATH = 'CollageDetectionModelWeights.h5'
id2label = {0: "single", 1: "collage"}

def srm_filter(img_path, target_size=(224, 224)):
  img = cv2.imread(img_path)
  dst = cv2.filter2D(img, -1, filter1)
  dst = cv2.filter2D(dst, -1, filter2)
  dst = cv2.filter2D(dst, -1, filter3)
  dst = cv2.resize(dst, target_size, interpolation = cv2.INTER_AREA)
  dst = dst.reshape((1, dst.shape[0], dst.shape[1], dst.shape[2]))

  return dst

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Path to the image")
    ap.add_argument("-c", "--checkpoint", required=False, help="Path to the pre-trained model")
    args = vars(ap.parse_args())

    if 'checkpoint' not in args:
        weights_path = WEIGHTS_PATH
    else:
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

    # inferencing
    image = srm_filter(args["image"])
    predicted_id = np.argmax(model.predict(image)[0])
    print(f"Predicted label: {id2label[predicted_id]}")
