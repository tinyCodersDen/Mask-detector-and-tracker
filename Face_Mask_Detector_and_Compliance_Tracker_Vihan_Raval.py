import cv2
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.animation as animation
import multiprocessing
import threading
import time
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  
wearing=[]
not_wearing=[]
count_w=0
count_n=0
no_face=0
cap=cv2.VideoCapture(0)  
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)  
style.use('fivethirtyeight')
def plot():
    # global count
    def animate(i):
        wearing.append(count_w)
        not_wearing.append(count_n)
        ax1.clear()
        ax1.plot(wearing)
        ax1.plot(not_wearing)
        ax1.set_yticks(np.arange(0,max([count_w,count_n]),1),minor='False')
        ax1.set_xticks(np.arange(0,wearing[-1]+not_wearing[-1]+no_face,1),minor='False')
        # print(np.arange(0,wearing[-1]+not_wearing[-1]+no_face,1),minor='False')
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
def detection():
    global count_w
    global count_n
    list1=[]
    # def calc():
    #     start = time.time()
    #     for i in range(0, 50) :
    #         ret, frame = cap.read()
    #     end = time.time()
    while True:
        ret,img=cap.read()
        img = cv2.flip(img,1)
        cv2.rectangle(img,(0,0),(700,10),(0,0,0),50)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
        (thresh, black_and_white) = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)
        eyes = eye_cascade.detectMultiScale(gray)
        if (len(eyes) == 0 and len(faces) == 0):
            cv2.putText(img, "No Face Detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            list1.append('No one')
        elif(len(faces) == 0 and len(faces_bw) == 1):
            # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
            cv2.putText(img, "Mask Detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            list1.append("Mask")
        else:
            try:
                for (faceX,faceY,faceW,faceH) in faces: 
                    cv2.rectangle(img,(faceX,faceY),(faceX+faceW,faceY+faceH),(255,255,0),2)  
                    roi_gray = gray[faceY:faceY+faceH, faceX:faceX+faceW] 
                    roi_color = img[faceY:faceY+faceH, faceX:faceX+faceW] 
                    for (eyeX,eyeY,eyeW,eyeH) in eyes: 
                        cv2.rectangle(roi_color,(eyeX,eyeY),(eyeX+eyeW,eyeY+eyeH),(0,127,255),2) 
                    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
                if len(mouth_rects) == 0:
                        cv2.putText(img, "Mask Detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
                        list1.append("Mask")
                else:
                        for (mx, my, mw, mh) in mouth_rects:

                            if(faceY < my < faceY + faceH):
                                cv2.putText(img, "No Mask Detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                                list1.append("No Mask")
                                #cv2.rectangle(img, (mx, my), (mx + mh, my + mw), (0, 0, 255), 3)
                                break
            except UnboundLocalError:
                print('hello')
        # print(len(list1))                
        if len(list1)>=5:
            string1=list1[0]
            for x in list1:
                if x==string1:
                    pass
                else:
                    list1=[]
                    string1=''
            if string1=='Mask':
                count_w+=1
            elif string1=='No Mask':
                count_n+=1
            elif string1=='No one':
                no_face+=1
            list1=[]
        cv2.imshow("Mask Detection",img)
        if cv2.waitKey(1)==ord('q'):
            break
thread1=threading.Thread(target=detection)
thread1.start()
plot()
cap.release()
cv2.destroyAllWindows()