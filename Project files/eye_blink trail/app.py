from flask import Flask, render_template,Response

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time
import cv2
app=Flask(__name__)
camera = cv2.VideoCapture(0)
def gen_frames():
     while True:
         success, frame=camera.read()
         if not success:
             break
         else:
             ret,buffer=cv2.imencode('.jpg',frame)
             frame=buffer.tobytes()
         yield(b'--frame\r\n'b'Content-type:image/jpeg\r\n\r\n'+frame+b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video')
def video():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')



