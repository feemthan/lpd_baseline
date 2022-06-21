import os
from easyocr import Reader
import cv2
import json
import numpy as np

import pdb
from detectron2.engine import DefaultPredictor

import os
import pickle


dataset_root = "license-plate-dataset"
output_root = "output"

cfg_save_path = output_root + "/anpd/od_cfg.pickle"
with open(cfg_save_path, "rb") as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

lpd_pred = DefaultPredictor(cfg)

# on_video(video_path, predictor)

reader = Reader(['en'])


# def recognizer(image_name, cropped_image, bbox, output_folder, actual_image):

#     out_text = reader.readtext(cropped_image)
#     # pdb.set_trace()
#     if out_text != []:
#         output=''.join([i[1] for i in out_text])
#         score = [i[2] for i in out_text]
#         actual_score = (sum(score)/len(score))
#         if output!=[]:
#             _ = actual_image
#             x, y, w, h = bbox
#             _ = cv2.rectangle(_, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             _ = cv2.putText(_, output, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#             _ = cv2.putText(_, str(round(actual_score, 2)), (x+w-40, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#             cv2.imwrite(output_folder + '/' + image_name, cv2.cvtColor(_, cv2.COLOR_BGR2RGB))


def recog_v3(detected_image, actual_image, detected_bbox, image_name, output_folder):
    outputs = lpd_pred(detected_image)
    # viz = Visualizer(
    #     im[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION
    # )
    # viz = viz.draw_instance_predictions(outputs["instances"].to("cpu"))
    # pdb.set_trace()
    bbox = outputs['instances'].pred_boxes.tensor.cpu().numpy().astype(int).tolist()
    if bbox == []:
        return
    x1, y1, x2, y2 = bbox[0]
    # plt.figure(figsize=(14, 10))
    # plt.imshow(viz.get_image())
    # boxes = outputs["instances"].pred_boxes
    # box = list(boxes)[0].detach().cpu().numpy().astype(int)
    # box = x1, y1, x2, y2
    # pdb.set_trace()

    # [y:y+h, x:x+w]

    out_text = reader.readtext(detected_image[y1:y2, x1:x2])
    if out_text == []:
        out_text = reader.readtext(detected_image)
    if out_text!=[]:
        output=''.join([i[1] for i in out_text])
        score = [i[2] for i in out_text]
        actual_score = (sum(score)/len(score))
        if output!=[]:
            _ = actual_image
            x, y, w, h = detected_bbox
            _ = cv2.rectangle(_, (x, y), (x+w, y+h), (255, 0, 0), 2)
            _ = cv2.putText(_, output, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            _ = cv2.putText(_, str(round(actual_score, 2)), (x+w-40, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imwrite(output_folder + '/' + image_name, cv2.cvtColor(_, cv2.COLOR_BGR2RGB))
