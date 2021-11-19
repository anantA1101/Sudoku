import numpy as np
import cv2

# now we create functions for feature extractors 

# fist function : detecting face and returning cropped face if no face detected then returns the input face 
def num_extractor(imgg):
    cropped = imgg[150:650 , 700:1200]

    return cropped 
    

#initializing web cam 0: if internal web cam; 1: if exteral webcam 

cap= cv2.VideoCapture(0)
address="http://192.168.4.161:8080/video"
cap.open(address)

count = 400

# now we read the camera input and store it in return frame as shown 

while True:
    ret,frame=cap.read()                                             
    count +=1       
    numb =cv2.resize(num_extractor(frame),(400,400))                
    frame_bw= cv2.cvtColor(numb,cv2.COLOR_BGR2GRAY)
    (thresh, frame_BW)= cv2.threshold(frame_bw,153,255,cv2.THRESH_BINARY)
       

    file_name_path= r'D:\projects\numbers_dataset\9/' + str(count) + '.jpg' 
    cv2.imwrite(file_name_path,frame_BW)

        
    cv2.putText(frame_BW ,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)          # to put a text on the photo   
    cv2.imshow('num Cropper', frame_BW )                                                    # this commad is to display the face to us 


    key =cv2.waitKey(40)

    if key== 27 or count ==500:               # this is when we want to stop the program that is when we press 'enter' which has the code 13 or when there are 100 images taken  
        break
 
cap.release()                                           #This is to reease the webcam  
cv2.destroyAllWindows()                                 # this is a safety command to close the windows later 
print ("collecting Samples Complete ")             
     