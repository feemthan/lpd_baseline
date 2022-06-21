import pathlib
from detect import detect_vehicles
# from recog_v2 import recognizer
import subprocess

directory = 'cctv'
output_folder = 'lpd_output'
# class_id = [2]
class_ids = [1, 2, 3, 5, 7]
# class_ids = [3]
detection_score_threshold = 0.90
recog=True

for filepath in pathlib.Path(directory).glob('**/*'):
    image_path = str(filepath.absolute())
    image_name = str(filepath.name)
    detect_vehicles(image_path, image_name, class_ids, output_folder, recog, detection_score_threshold)

subprocess.run(["ffmpeg", "-framerate 30", "-i frame%d.jpg", "output.mp4"])