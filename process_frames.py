import os
import pathlib
from detect import detect_vehicles
from recog_v2 import recognizer

directory = 'cctv'
output_folder = 'lpd_output'
# class_id = [2]
class_ids = [1, 2, 3, 5, 7]
# class_ids = [3]
score_threshold = 0.90
recog=True

for filepath in pathlib.Path(directory).glob('**/*'):
    image_path = str(filepath.absolute())
    image_name = str(filepath.name)
    detect_vehicles(image_path, image_name, class_ids, output_folder, score_threshold, recog)
