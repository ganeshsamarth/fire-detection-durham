''' Used for labelling, outputs a frame for every second
    in the video into a directory '''

import cv2
import numpy as np
import sys
import os


read_directory = '/home/pbu/Desktop/fire-detection-software'

print ("\nSearching {} for mp4 files..\n".format(read_directory))

blacklist = ["2ndAlarmFire117N.mp4", "CarInFlames-FireFighterHelmetCam.mp4", "WaterMistFireDemonstration.mp4"]

for filename in os.listdir(read_directory):

    if '.mp4' in filename and filename not in blacklist:

        name = filename[:-4]

        os.makedirs("{}/{}".format(read_directory, name))
        write_directory = "{}/{}".format(read_directory, name)
        video = cv2.VideoCapture(os.path.join(read_directory, filename))
        fps = int(video.get(5)/2)
        total_frames = video.get(7)
        count = 0
        

        while True:

            ret, frame = video.read()
            print('processing')

            if not ret:
                break

            if count % fps == 0:

                smallFrame = cv2.resize(frame, (224,224), cv2.INTER_AREA)
                cv2.imwrite("%s%s%d.png" % (write_directory, name, count), smallFrame)

            count += 1
            

        video.release()

print()
