import os
from easyocr import Reader
import cv2
import json
import numpy as np

import pdb

reader = Reader(['en'])


def recognizer(image_name, cropped_image, bbox, output_folder, actual_image):
    # image = cv2.resize(cropped_image, (800, 600))

    # gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) 
    # blur = cv2.GaussianBlur(gray, (3,3), 0)
    # edged = cv2.Canny(blur, 10, 200) 

    out_text = reader.readtext(cropped_image)
    # pdb.set_trace()
    if out_text != []:
        output=''.join([i[1] for i in out_text])
        score = [i[2] for i in out_text]
        actual_score = (sum(score)/len(score))
        if output!=[]:
            _ = actual_image
            x, y, w, h = bbox
            _ = cv2.rectangle(_, (x, y), (x+w, y+h), (255, 0, 0), 2)
            _ = cv2.putText(_, output, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            _ = cv2.putText(_, str(round(actual_score, 2)), (x+w-40, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imwrite(output_folder + '/' + image_name, cv2.cvtColor(_, cv2.COLOR_BGR2RGB))
