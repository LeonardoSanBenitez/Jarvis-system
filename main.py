# Math and ML
import cv2
import numpy as np

# hardware, sensors and actuators
import board
import digitalio
from picamera import PiCamera
import sensor_bme280

# Utils
import time

class Flags:
    exportLog = False
    imgFromDisk = False
    classifierModel = 'models/haarcascade_frontalface_default.xml'


class Camera ():
    def __init__(self, imgWidht=800, imgHeight=800):
        self.imgWidht = imgWidht
        self.imgHeight = imgHeight
        self.camera = PiCamera()
        self.camera.resolution = (imgWidht, imgHeight)
        self.camera.framerate = 15
        self.camera.rotation = 90
        time.sleep(5)
        self.camera.start_preview()

    def captureFrame(self, verbose=True):
        if verbose:
            print("begin readCamera...", end='')
        image = np.empty((self.imgHeight * self.imgWidht * 3,), dtype=np.uint8)
        self.camera.capture(image, 'bgr')
        image = image.reshape((self.imgHeight, self.imgWidht, 3))
        if verbose:
            print("...Finishing readCamera")
        return image

    def __del__(self):
        self.camera.stop_preview()

if __name__ == "__main__":
    flags = Flags()
    camera = Camera()
    bme280 = sensor_bme280.Bme280()
    cascade = cv2.CascadeClassifier(flags.classifierModel)
    
    i=0

    print("Begining test")
    while(True):
        # Camera
        if flags.imgFromDisk:
            frame = cv2.imread("sample_img.jpg")
        else:
            frame = camera.captureFrame()
        print(frame.shape)
        print(frame.mean())
        
        cv2.imwrite('tempImgs/img' + str(i) + '.jpg', frame)
        print('saved ', i, 'image')
        i+=1

        # Sensors
        #print(bme280.readBME280All())

        # Image "machine learning"
        #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #objects = cascade.detectMultiScale(gray, 1.3, 5)
        #print(objects)

        print('---------')
