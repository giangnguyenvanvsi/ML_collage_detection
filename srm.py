
import numpy as np
import cv2

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

def srm_filter(img, target_size=(224, 224)):
  dst = cv2.filter2D(img, -1, filter1)
  dst = cv2.filter2D(dst, -1, filter2)
  dst = cv2.filter2D(dst, -1, filter3)
  dst = cv2.resize(dst, target_size, interpolation = cv2.INTER_AREA)
  dst = dst.reshape((1, dst.shape[0], dst.shape[1], dst.shape[2]))
  return dst













