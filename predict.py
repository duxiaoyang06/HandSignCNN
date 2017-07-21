import tensorflow as tf
import os
import glob
import numpy as np
import cv2


sess=tf.Session()    
saver = tf.train.import_meta_graph('my_test_model_iteration_310.meta')
saver.restore(sess,'my_test_model_iteration_310')


graph = tf.get_default_graph()

x = graph.get_tensor_by_name("x:0")
y_conv = graph.get_tensor_by_name("y_conv:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")
y_conv_cls = graph.get_tensor_by_name("y_conv_cls:0")

img_size = 52
test_path ="testing_data/C"
images = []
path = os.path.join(test_path, '*g')
print("path")
print(path)
files = glob.glob(path)
for fl in files:
	image = cv2.imread(fl)
	image = cv2.resize(image, (img_size, img_size), cv2.INTER_LINEAR)
	images.append(image)

images = np.array(images)
train_batch_size = 5
img_size_flat = img_size * img_size * 3
x_batch = images;
x_batch = x_batch.reshape(train_batch_size, img_size_flat)
feed_dict = {x:x_batch,keep_prob:1}


print sess.run(y_conv,feed_dict)
print sess.run(y_conv_cls,feed_dict)