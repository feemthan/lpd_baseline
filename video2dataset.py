import cv2
vidcap = cv2.VideoCapture('vid3.mp4')
success,image = vidcap.read()
count = 0
while success:
    try:
        cv2.imwrite("cctv_small/frame%d.jpg" % count, image)
        cv2.imwrite("lpd_output_small/frame%d.jpg" % count, image)
        success,image = vidcap.read()
        count += 1
    except:
        break