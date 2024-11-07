import cv2
import dlib
import sys
from scipy.spatial import distance
from imutils import face_utils
import face_recognition
import random
s=input()
'''It's a landmark's facial detector with pre-trained models, the dlib is used to estimate the location of 68 coordinates (x, y) that map the facial points on a person's face like image below.'''

def reset():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cv2.imwrite('Me.jpg', frame)
    # cv2.destroyAllWindows()
    cap.release()
if s == 'reset':
    reset()
def take_picture():
    cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cv2.imwrite('picture.jpg', frame)
    #cv2.destroyAllWindows()
    cap.release()

def analyze_user():

    baseimg=face_recognition.load_image_file('Me.jpg')
    baseimg=cv2.cvtColor(baseimg, cv2.COLOR_BGR2RGB)
    myface=face_recognition.face_locations(baseimg)[0]
    encodemyface=face_recognition.face_encodings(baseimg)[0]
    #cv2.rectangle(baseimg,(myface[3],myface[0]),(myface[1],myface[2]),(255,0,255),2)
    #cv2.imshow("Test", baseimg)
    #cv2.waitKey(0)

    sampleimg = face_recognition.load_image_file("Picture.jpg")
    sampleimg = cv2.cvtColor(sampleimg, cv2.COLOR_BGR2RGB)

    samplefacetest=face_recognition.face_locations(sampleimg)[0]
    try:
        encodesamplefacetest = face_recognition.face_encodings(sampleimg)[0]
    except IndexError as e:
        print("Index Error....Authentication failed")
        sys.exit()

    cv2.rectangle(sampleimg,(samplefacetest[3],samplefacetest[0]),(samplefacetest[1],samplefacetest[2]),(255,0,255),2)
    cv2.imshow("Test",sampleimg)

    result=face_recognition.compare_faces([encodemyface],encodesamplefacetest)
    resultstring=str(result)
    if resultstring == '[True]':
        print("Face authentication successful")
        print("Now blink your eyes ")
        Mpass = '2233'
        cap = cv2.VideoCapture(0)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        def eye_aspect_ratio(eye):
            A = distance.euclidean(eye[1], eye[5])

            B = distance.euclidean(eye[2], eye[4])

            C = distance.euclidean(eye[0], eye[3])
            eye = (A + B) / (2.0 * C)

            return eye

        count = 0
        total = 0
        password = ''

        while True:
            success, img = cap.read()
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(imgGray)

            for face in faces:
                landmarks = predictor(imgGray, face)

                landmarks = face_utils.shape_to_np(landmarks)
                leftEye = landmarks[42:48]
                rightEye = landmarks[36:42]

                leftEye = eye_aspect_ratio(leftEye)
                rightEye = eye_aspect_ratio(rightEye)

                eye = (leftEye + rightEye) / 2.0

                if eye < 0.3:
                    count += 1
                else:
                    if count >= 3:
                        total += 1

                    count = 0
            cv2.putText(img, "Blink Count: {}".format(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Video', img)

            if cv2.waitKey(1) & 0xff == ord(' '):
                password += str(total)
                total = 0
            elif cv2.waitKey(1) & 0xff == ord('q'):
                break
            if len(password) == 4:
                print(password)
                break

        if password == Mpass:
            print("successfully logged in")
        else:
            print("Try again")

    else:
        print("In valid user")
        l=list(range(0,10))
        ran_pass=''
        for i in range(0,4):
            k=random.choice(l)
            ran_pass+=str(k)
        print(ran_pass)


take_picture()
analyze_user()

