import subprocess

subprocess.run(["ffmpeg", "-framerate", "30", "-i", "lpd_output/frame%d.jpg", "output.mp4"])