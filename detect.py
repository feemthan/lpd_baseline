from email.mime import image
import detectron2
import cv2
import json
import pdb

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
from recog_v2 import recognizer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS_SCORE_THRESH_TEST = 0.6
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

predictor = DefaultPredictor(cfg)


def detect_multiclass(image, instant, class_id, image_name, output_folder, recog=True):

    i = 0
    instant = instant[instant.pred_classes==class_id]

    for box in instant.pred_boxes.to('cpu'):
        x, y, x1, y1 = box.detach().numpy()
        w = x1 - x
        h = y1 - y
        x, y, w, h = x.astype(int), y.astype(int), w.astype(int), h.astype(int)

        detected_image = image[y:y+h, x:x+w]
        # image_new_name = image_name.split('.')[0] + '_' + str(i)
        # cv2.imwrite('crops/'+ image_new_name +'.jpg', cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))

        bbox=[x, y, w, h]
        # visualizer.draw_box(box)
        # v.draw_text(str(box[:2].numpy()), tuple(box[:2].numpy()))
        i+=1
        if recog==True:
            recognizer(image_name, detected_image, bbox, output_folder, actual_image=image)

def detect_vehicles(image_path, image_name, class_ids, output_folder, recog=True, score_threshold=0.95):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    outputs = predictor(image)
    # visualizer = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    instances = outputs['instances']
    instant = instances[instances.scores>score_threshold]
    for class_id in class_ids:
        # out = visualizer.draw_instance_predictions(instances[instances.pred_classes==class_id].to('cpu'))
        # import pdb
        # pdb.set_trace()
        # output_image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
        # cv2.imwrite('detections/'+ image_name, output_image)

        detect_multiclass(image, instant, class_id, image_name, output_folder, recog=recog)
    # return output_image
