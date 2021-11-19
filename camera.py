from keras.preprocessing import image 
from keras.models import load_model
import matplotlib.pyplot as plt 
import tensorflow as tf
from PIL import Image 
import numpy as np 
import cv2



#L O A D I N G  M O D E L

model= load_model(r'D:\projects\numbers_dataset\num_model.model')

#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************
#****************************************************   F U N T I O N S    *********************************************************************************
#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************

# F U N C T I O N   F O R   T R A C K E R 
def nothing():
    pass


#F U C N T I O N   F O R    A I    M O D E L   
def number_identification(img_to_identify):

    img_3d= cv2.cvtColor(img_to_identify, cv2.COLOR_GRAY2BGR) 
    img = cv2.resize(img_3d,(200,200))

    N_image= tf.keras.utils.normalize(img, axis=1)

    X=image.img_to_array(N_image)
    X= np.expand_dims(X,axis=0)
    images= np.vstack([X])

    val=model.predict(images)
    number = np.argmax(val)

    return number 
    

# F U N C T I O N   T O    C R O P   O U T   A   N U M B E R    F R O M    F R A M E 
def number_extractor(in_f,x_c,y_c,w_c,h_c):
    
    cropped= in_f[y_c: y_c+60, x_c : x_c+w_c]
    cv2.imshow("cropped" , cropped)

    return cropped



#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************
#*********************************************************C R E A T I N G    A   T R A C K E R **************************************************************************************************
#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************


#cv2.namedWindow("Tracking")

cv2.createTrackbar("LH", "Tracking", 0,255, nothing)
cv2.createTrackbar("LS", "Tracking", 0,255, nothing)
cv2.createTrackbar("LV", "Tracking", 0,255, nothing)
cv2.createTrackbar("UH", "Tracking", 255,255, nothing)
cv2.createTrackbar("US", "Tracking", 255,255, nothing)
cv2.createTrackbar("UV", "Tracking", 255,255, nothing)

    
#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************
#***************************************************************** C A M E R A ******************************************************************************************
#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************




# S T A R T I N G    C A M E R A   P H O N E    
cap= cv2.VideoCapture(0)
address="http://IP/video"
cap.open(address)


while True:
    _,frame= cap.read()
#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************

    hsv= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h  = cv2.getTrackbarPos("LH", "Tracking")                               
    l_s  = cv2.getTrackbarPos("LS", "Tracking")                                   
    l_v  = cv2.getTrackbarPos("LV", "Tracking")

    u_h  = cv2.getTrackbarPos("UH", "Tracking")
    u_s  = cv2.getTrackbarPos("US", "Tracking")
    u_v  = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([92, 0 , 0])
    u_b = np.array([129, 118 , 232])

#***********************************************************************************************************************************************************
#***********************************************************************************************************************************************************

# C O L O R     O P T I M I Z A T I O N    F O R    B E T T E R    U T I L I S A T I O N     O F   I M A G E 

    frame_bw= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    (thresh, frame_BW)= cv2.threshold(frame_bw,153,255,cv2.THRESH_BINARY)
    
    contours,_ = cv2.findContours(frame_BW, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    


    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>320 and area<1500:
            cv2.drawContours(frame,[cnt],-1,(0,0,255))
            x,y,w,h= cv2.boundingRect(cnt)
            

            if x!= 0 and y!=0 and w!= 0 and h!= 0 and w*h> 700 :
                cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0,3))

                number =number_extractor(frame_BW,x,y,w,h)
                
                num_value= number_identification(number)

                cv2.putText(frame,str(num_value),(x,y),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),1)
            
            
    

    cv2.imshow("frame",frame)                
    cv2.imshow("Black and White", frame_BW)
    
    

    key= cv2.waitKey(40)
    if key==27:
        break


cap.release()
cv2.destroyAllWindows

