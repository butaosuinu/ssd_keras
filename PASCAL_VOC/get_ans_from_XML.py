import numpy as np
import os
from xml.etree import ElementTree

class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.num_classes = 20
        self.data = dict()
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            bounding_boxes = []
            one_hot_classes = []
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = int(bounding_box.find('xmin').text)
                    ymin = int(bounding_box.find('ymin').text)
                    bw = int(bounding_box.find('xmax').text) - xmin
                    bh = int(bounding_box.find('ymax').text) - ymin
                bounding_box = [xmin,ymin,bw,bh]
                bounding_boxes.append(bounding_box)
                class_name = object_tree.find('name').text
                # one_hot_class = self._to_one_hot(class_name)
                one_hot_classes.append([class_name])
            image_name = root.find('filename').text
            # bounding_boxes = np.array(bounding_boxes)
            # one_hot_classes = np.array(one_hot_classes)
            print(one_hot_classes)
            print(bounding_boxes)
            image_data = np.hstack((one_hot_classes, bounding_boxes))
            self.data[image_name] = image_data

## example on how to use it
import pickle
data = XML_preprocessor('test_img/anno/').data 
pickle.dump(data,open('my_test_ans.pkl','wb'))

