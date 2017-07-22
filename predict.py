import tensorflow as tf
import os
import glob
import numpy as np
import cv2
from hand_cnn import HandCnn

test_path ="testing_data/C"
cnn = HandCnn()
predictions = cnn.predict_batch(test_path)
print(predictions)