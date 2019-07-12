import cv2
import sys
import numpy as np

class CaptureCamera:
    def __init__(self):
        CAS_NAME = './cascade.xml'
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(CAS_NAME)

    def run(self,mirror=True,size=None):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            face_flag = False
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces  =  self.face_cascade.detectMultiScale(gray, minNeighbors=20)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                face_flag = True
            if size is not None and len(size) == 2:
                  frame = cv2.resize(frame, size)
            cv2.imshow('camera capture', frame)
            key = cv2.waitKey(1)
            if key == 27: # ESCキーで終了
                break

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = CaptureCamera()
    camera.run()
