import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf

import glob
import json

from ssd import SSD300
from ssd_utils import BBoxUtility

# %matplotlib inline
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
voc_classes = ['Title', 'Author', 'Num', 'Sub_title', 'Series_name',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1

network_size = 300
input_shape=(network_size, network_size, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights.50-firstTime.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)
inputs = []
images = []

def get_image_from_path(img_path):
    img = image.load_img(img_path, target_size=(network_size, network_size))
    img = image.img_to_array(img)
    images.append([imread(img_path), img_path])
    inputs.append(img.copy())

img_list = glob.glob('test_img/*.jpg')
for ig in img_list:
    get_image_from_path(ig)

inputs = preprocess_input(np.array(inputs))
preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)
# %%time
a = model.predict(inputs, batch_size=1)
b = bbox_util.detection_out(preds)

sub_bbs_arr = []

for i, img in enumerate(images):
    sub_bb = []
    # print(img[1])
    # print(results[i])
    # Parse the outputs.
    if [] != results[i]:
        det_label = results[i][:, 0]
        det_conf = results[i][:, 1]
        det_xmin = results[i][:, 2]
        det_ymin = results[i][:, 3]
        det_xmax = results[i][:, 4]
        det_ymax = results[i][:, 5]
    else :
        det_label = [0]
        det_conf = [0]
        det_xmin = [0]
        det_ymin = [0]
        det_xmax = [0]
        det_ymax = [0]

    # Get detections with confidence higher than 0.6.
    # top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    top_indices = [i for i, conf in enumerate(det_conf) if conf > 0]

    if [] != top_indices:
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    plt.imshow(img[0] / 255.)
    currentAxis = plt.gca()

    if [] != top_indices:
        for i in range(top_conf.shape[0]):
            xmin = int(round(top_xmin[i] * img[0].shape[1]))
            ymin = int(round(top_ymin[i] * img[0].shape[0]))
            xmax = int(round(top_xmax[i] * img[0].shape[1]))
            ymax = int(round(top_ymax[i] * img[0].shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = voc_classes[label - 1]
            display_txt = '{:0.2f}, {}'.format(score, label_name)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            color = colors[label]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
            
            # print([img[1], label_name, score, xmin, ymin, xmax - xmin, ymax - ymin])
            sub_bb.append([img[1], label_name, score, xmin, ymin, xmax - xmin, ymax - ymin])

    # print(sub_bb)
    sub_bbs_arr.extend(sub_bb)

    # plt.show()

# print(sub_bbs_arr)
with open('sub_bbs.json', 'w') as fw:
    json.dump(sub_bbs_arr, fw, ensure_ascii=False, indent=4)
