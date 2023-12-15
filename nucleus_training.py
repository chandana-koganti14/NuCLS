import os
from numpy import zeros, asarray
import pandas as pd
import numpy as np
import mrcnn.utils
import mrcnn.config
import mrcnn.model
import skimage
from skimage.io import imread
import csv
import scipy
import matplotlib.pyplot as plt
import cv2
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from skimage.io import imread
import skimage.color

class NucleusDataset(mrcnn.utils.Dataset):

    
    def load_dataset(self, dataset_dir, is_train=True): 
        self.add_class("nucleus", 1, "tumor")
        self.add_class("nucleus", 2, "fibroblast")
        self.add_class("nucleus", 3, "lymphocyte")
        self.add_class("nucleus", 4, "plasma_cell")
        self.add_class("nucleus", 5, "macrophage")
        self.add_class("nucleus", 6, "mitotic_figure")
        self.add_class("nucleus", 7, "vascular_endothelium")
        self.add_class("nucleus", 8, "myoepithelium")
        self.add_class("nucleus", 9, "apoptotic_body")
        self.add_class("nucleus", 10, "neutrophil")
        self.add_class("nucleus", 11, "ductal_epithelium")
        self.add_class("nucleus", 12, "eosinophil")
        self.add_class("nucleus", 13, "unlabeled")
        

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.csv'
           
            self.add_image(source="nucleus", image_id=image_id, path=img_path, annot_path=ann_path, class_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13])
    
    def load_mask(self, image_id):
        
        info = self.image_info[image_id] 
        path = info['annot_path']
        
        boxes, image_dims = self.extract_boxes(path) 

        h, w = image_dims[0]  
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]

            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if (box[4] == 'tumor'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('tumor'))
            elif(box[4] == 'fibroblast'):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('fibroblast')) 
            elif(box[4] == 'lymphocyte'):
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('lymphocyte'))
            elif(box[4] == 'plasma_cell'):
                masks[row_s:row_e, col_s:col_e, i] = 4
                class_ids.append(self.class_names.index('plasma_cell'))
            elif(box[4] == 'macrophage'):
                masks[row_s:row_e, col_s:col_e, i] = 5
                class_ids.append(self.class_names.index('macrophage'))
            elif(box[4] == 'mitotic_figure'):
                masks[row_s:row_e, col_s:col_e, i] = 6
                class_ids.append(self.class_names.index('mitotic_figure'))
            elif(box[4] == 'vascular_endothelium'):
                masks[row_s:row_e, col_s:col_e, i] = 7
                class_ids.append(self.class_names.index('vascular_endothelium'))
            elif(box[4] == 'myoepithelium'):
                masks[row_s:row_e, col_s:col_e, i] = 8
                class_ids.append(self.class_names.index('myoepithelium'))
            elif(box[4] == 'apoptotic_body'):
                masks[row_s:row_e, col_s:col_e, i] = 9
                class_ids.append(self.class_names.index('apoptotic_body'))
            elif(box[4] == 'neutrophil'):
                masks[row_s:row_e, col_s:col_e, i] = 10
                class_ids.append(self.class_names.index('neutrophil'))
            elif(box[4] == 'ductal_epithelium'):
                masks[row_s:row_e, col_s:col_e, i] = 11
                class_ids.append(self.class_names.index('ductal_epithelium')) 
            elif(box[4] == 'eosinophil'):
                masks[row_s:row_e, col_s:col_e, i] = 12
                class_ids.append(self.class_names.index('eosinophil'))            
            elif(box[4] == 'unlabeled'):
                masks[row_s:row_e, col_s:col_e, i] = 13
                class_ids.append(self.class_names.index('unlabeled'))    
        return masks, asarray(class_ids, dtype='int32')
        
    import csv

    def extract_boxes(self, filename):
        boxes = []
        image_dims = []
    
        csv_basename = os.path.basename(filename).split(".")[0]
        
        image_dir = 'nucleus' + '/images/'

        image_path = os.path.join(image_dir, csv_basename + ".png")
        
        image = imread(image_path)
        h, w = image.shape[:2]
        
        with open(filename) as f:

            reader = csv.reader(f)
            next(reader)
            for row in reader:
                raw=str(row[1])
                xmin = int(row[5]) 
                ymin = int(row[6])
                xmax = int(row[7])
                ymax = int(row[8])
                coors = [xmin, ymin, xmax, ymax,raw]
                boxes.append(coors)
        return boxes, image_dims
    def get_class_counts(self):
        class_counts = {class_info['id']: 0 for class_info in self.class_info}
        for image_id in self.image_ids:
            _, class_ids = self.load_mask(image_id)
            for class_id in class_ids:
                class_counts[class_id] += 1
        return class_counts

    def compute_class_weights(self, class_counts):
        total = sum(class_counts.values())  
        class_weights = {class_id : total/count if count > 0 else 0 for class_id, count in class_counts.items()}
        return class_weights
    

class NucleusConfig(mrcnn.config.Config):
    NAME = "nucleus_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 14

    STEPS_PER_EPOCH = 50
    def __init__(self, class_weights=None):
        super().__init__()
        self.CLASS_WEIGHTS = class_weights
train_dataset = NucleusDataset()
train_dataset.load_dataset(dataset_dir='nucleus', is_train=True)
train_dataset.prepare()
validation_dataset = NucleusDataset()
validation_dataset.load_dataset(dataset_dir='nucleus', is_train=False)
validation_dataset.prepare()
class_counts = train_dataset.get_class_counts()
class_weights = train_dataset.compute_class_weights(class_counts)
class_names = train_dataset.class_names
class_labels = [class_names[class_id] for class_id in class_weights.keys()]
weights = list(class_weights.values())

plt.figure(figsize=(12, 6))
bars = plt.bar(class_labels, list(class_weights.values()), color='blue')
plt.xticks(rotation=45, ha='right')
plt.title('Class Weights for Balancing')
plt.xlabel('Class')
plt.ylabel('Weight')
plt.tight_layout()
plt.show()
nucleus_config = NucleusConfig(class_weights=class_weights)

model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=nucleus_config)


model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])


model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 

            learning_rate=nucleus_config.LEARNING_RATE, 
            epochs=2, 
            layers='heads')

model_path = 'Nucleus_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
