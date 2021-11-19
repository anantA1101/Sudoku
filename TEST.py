import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import cv2

#loading the saved model from MNIST.py
new_model = tf.keras.models.load_model(r"D:\DEEP LEARNING CODES\Skuduko_AI\digit_model.model")
mnist = tf.keras.datasets.mnist

#To check if the data and priction is right
#loading the mnist data and dividing into test and train 
(x_train,y_train) , (x_test, y_test)= mnist.load_data()

#normalising the imag values from 255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)



#pridicting the output using test images

#predictions = new_model.predict()

img= cv2.imread("D:\projects\W.jpeg")
print(img.shape)

frame_bw= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("grayscale shape "+str(frame_bw.shape))

resizing= cv2.resize(frame_bw, (28,28))
print(resizing.shape)

N_image= tf.keras.utils.normalize(resizing, axis=1)
print(N_image.shape)

    
img_array=np.array(N_image)
img_array=np.expand_dims(img_array,axis=0)

pred=new_model.predict(img_array)
print("probablities "+str(pred))
print(np.argmax(pred))

#plotiing the actual image of the number 

plt.imshow(N_image, cmap=plt.cm.binary)
plt.show()

plt.imshow(x_test[2], cmap=plt.cm.binary)
plt.show()
#printing the pridiction made 
""" print(np.argmax(predictions[3])) """
