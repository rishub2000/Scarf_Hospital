import cv2
import numpy as np
import os
from train_v2 import main

pic_no=0
print('enter the name of the person for enrollment')
name=input()
if not os.path.exists('./Faces/'+name):
    os.makedirs('./Faces/'+name)
fa=cv2.CascadeClassifier('faces.xml')
cap=cv2.VideoCapture(2)
ret=True
while ret:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=fa.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        if x-50>=0 and y-50>=0 and x+w+50<=frame.shape[1] and y+h+50<=frame.shape[0]:
            cropped=frame[y-30:y+h+30,x-30:x+w+30]
            cv2.rectangle(frame,(x-50,y-50),(x+w+50,y+h+50),(255,0,0),2,cv2.LINE_AA)
            pic_no=pic_no+1
            cv2.imwrite('./Faces/'+name+'/'+str(pic_no)+'.jpg',cropped)
        else:
            pass
    cv2.imshow('frame',frame)
    cv2.waitKey(100)

    if(pic_no>50):
    	break


cap.release()
cv2.destroyAllWindows()
main()
