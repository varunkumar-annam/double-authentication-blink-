from flask import Flask, render_template,request,redirect,url_for,flash
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
import time
from flask_mysqldb import MySQL
import yaml
from tabulate import tabulate
import numpy as np
from os import listdir
from os.path import isfile, join


app=Flask(__name__)
app.secret_key = "super secret key"
db = yaml.safe_load(open('db.yaml'))
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)


face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
      return None

    for (x,y,w,h) in faces:
         cropped_face = img[y:y+h, x:x+w]
    return cropped_face



def face_detector(img, size=0.5):

   # Convert image to grayscale\n",
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, [],

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2),
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

def im_collector():
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = './faces/' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not found")
            pass
        if cv2.waitKey(1) == 13 or count == 20:
            break
    cap.release()
    cv2.destroyAllWindows()


def train_data():
    data_path = "./faces/"
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    Training_Data, Labels = [], []
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    return model


def face_check():
    model = train_data()
    cap = cv2.VideoCapture(0)
    while True:
        tm = time.time()
        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            print("madhan")
            # Pass face to prediction model\n",
            # \"results\" comprises of a tuple containing the label and the confidence value\n",
            results = model.predict(face)

            if results[1] < 500:
                confidence = int(100 * (1 - (results[1]) / 400))
                display_string = str(confidence) + '% Confident it is User'

            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 120, 150), 2)

            if confidence >= 85:
                print(confidence)
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                return 1
            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Recognition', image)
                print(confidence)
                return 2


        except:
            cv2.putText(image, "No Face Found", (220, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Recognition', image)
            print('image not found')
            return 0

def blink():
    cap = cv2.VideoCapture(0)

    detector = FaceMeshDetector(maxFaces=1)
    plotY = LivePlot(640, 360, [20, 50], invert=True)
    idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
    ratioList = []
    blinkCounter = 0
    counter = 0
    color = (255, 0, 255)
    delay_key = 0
    password1=open("pass.txt",'r')
    pass_word=password1.read()
    password1.close()
    fin_key = ''
    while True:
        tm = time.time()

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            for id in idList:
                tuple(cv2.circle(img, face[id], 5, color, cv2.FILLED))
            leftUp = face[159]
            leftDown = face[23]
            leftLeft = face[130]
            leftRight = face[243]
            lenghtVer, _ = detector.findDistance(leftUp, leftDown)
            lenghtHor, _ = detector.findDistance(leftLeft, leftRight)
            cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
            cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
            ratio = int((lenghtVer / lenghtHor) * 100)
            ratioList.append(ratio)
            if len(ratioList) > 3:
                ratioList.pop(0)
            ratioAvg = sum(ratioList) / len(ratioList)
            if ratioAvg < 35 and counter == 0:
                blinkCounter += 1
                delay_key = 0
                color = (0, 200, 0)
                counter = 1
            if counter != 0:
                counter += 1
                if counter > 10:
                    counter = 0
                    color = (0, 0, 255)

            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100),
                               colorR=color)
            imgPlot = plotY.update(ratioAvg, color)
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
        else:
            img = cv2.resize(img, (640, 360))
            imgStack = cvzone.stackImages([img, img], 2, 1)
        delay_key += 1647701038
        if delay_key == (30 * 1647701038):
            if len(fin_key) == 0 and blinkCounter == 0:
                pass
            else:
                fin_key += str(blinkCounter)
            blinkCounter = 0
        print(fin_key)
        if len(fin_key) == len(pass_word):
            break
        cv2.imshow("Image", imgStack)
        cv2.waitKey(25)
    print(tm)
    return fin_key




@app.route('/')
def index():
    return render_template("index.html")


@app.route('/reset')
def reset():
    face_class=face_check()
    if face_class==1:
        password1 = open("pass.txt", 'r')
        pass_word = password1.read()
        password1.close()
        if blink() == pass_word:
            im_collector()
            train_data()
            k=blink()
            f=open("pass.txt","w")
            f.write(k)
            f.close()
    flash("oops! something went wrong try again!")
    return render_template('dashboard.html')
@app.route('/register')
def register():
    # face_class=face_check()
    # if face_class==1:
    password1 = open("pass.txt", 'r')
    pass_word = password1.read()
    password1.close()
    # if blink() == pass_word:
    im_collector()
    train_data()
    # k=blink()
    # f=open("pass.txt","w")
    # f.write(k)
    # f.close()
    flash("oops! something went wrong try again!")
    return render_template('register.html')

@app.route('/auth')
def auth():
    var = face_check()
    print('madhan',var)
    if var==1:
        password1 = open("pass.txt", 'r')
        pass_word = password1.read()
        password1.close()
        if blink()==pass_word:
            return render_template("dashboard.html")
        else:
            return render_template('back.html')
    elif var == 0:
        return render_template('back.html')
    elif var == 2:
        return render_template('back.html')
    else:
        return render_template('back.html')


@app.route('/dash', methods=['GET', 'POST'])
def dash():
        return render_template('dashboard.html')





@app.route('/fill', methods=['GET', 'POST'])
def fill():

    if request.method == 'POST':
        # Fetch form data
        userDetails = request.form

        event = userDetails['event']
        dt = userDetails['dt']
        orga = userDetails['organizer']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO events(event, dt, organizer) VALUES(%s, %s,%s)",(event, dt, orga))
        mysql.connection.commit()
        cur.close()
        return render_template('succes.html')
    return render_template('fill.html')



@app.route('/data')
def users():
    cur = mysql.connection.cursor()
    resultValue = cur.execute("SELECT * FROM events")
    if resultValue > 0:
        userDetails = cur.fetchall()
        with open("info.txt", "w") as outputfile:
            outputfile.write(tabulate(userDetails))

        return render_template('data.html',userDetails=userDetails)



if __name__=="__main__":
    app.run(debug=True)