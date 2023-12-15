import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os



CLASS_NAMES = ["BG","tumor", "fibroblast", "lymphocyte", "plasma_cell", 
               "macrophage", "mitotic_figure", "vascular_endothelium",  
               "myoepithelium", "apoptotic_body", "neutrophil", 
               "ductal_epithelium", "eosinophil", "unlabeled"]

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)
model = mrcnn.model.MaskRCNN(mode="inference",
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

model.load_weights(filepath="Nucleus_mask_rcnn_trained-new.h5", 
                   by_name=True)


image = cv2.imread(r"C:\Users\ADMIN\Mask-RCNN-TF2\nucleus-transfer-learning\nucleus\images\TCGA-A2-A1G6-DX1.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
r = model.detect([image], verbose=0)
r = r[0]


mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
