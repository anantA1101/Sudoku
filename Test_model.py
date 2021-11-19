import cv2
from keras.models import load_model
from keras.preprocessing import image 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import os

model= load_model(r'D:\projects\numbers_dataset\num_model.model')


dir_path=r"D:\projects\numbers_dataset\testing"
#model.summary()
for i in os.listdir(dir_path):

    img = image.load_img(dir_path+"//"+i,target_size=(200,200,3))
    plt.imshow(img)
    plt.show()

    X=image.img_to_array(img)
    X= np.expand_dims(X,axis=0)
    images= np.vstack([X])
    

    val=model.predict(images)
    print (val)

