import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from  keras.preprocessing.image import img_to_array
from  keras.preprocessing.image import load_img
import keras
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
import paramiko
import time
def hand_write(angle,fg):
        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect('10.232.169.2',port=22,username='ubuntu',password='raspberry')
            stdin,stdout,stderr=ssh.exec_command("cd /run/shm/; > angles echo {0}".format(angle))
            stdin,stdout,stderr=ssh.exec_command("cd /run/shm/; > ev_on_off echo {0}".format(fg))

def move(num):
    fg = 0
    if num > 5:
        num =1
    with open('angle_{0}.txt'.format(int(num)),'r') as f:
         angles = f.readlines()
    sw = [0,0,0,1,1,1,1,0,0,0,0,0,0,0]
    for i,angle in enumerate(angles):
        print('step{0}'.format(i))
        fg = sw[i]
        hand_write(str(angle),fg)
        time.sleep(0.5)
                                                

def predict():
    model=Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    model.load_weights('mnist.h5')
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    return model

img_path='image.png'
if __name__=="__main__":
    old = 100
    capture = cv2.VideoCapture(0)
    
    if capture.isOpened() is False:
        print("IO Error")

    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

    while True:

        ret, image = capture.read()

        if ret == False:
            continue
        image =cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        x,y,w,h= cv2.boundingRect(image)
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),2)
        rt, bw = cv2.threshold(image, 50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bw = cv2.bitwise_not(bw)
        bw = cv2.rectangle(bw,(160,120),(480,360),(255,255,255),5)
        cv2.imshow("Capture", bw)
        k = cv2.waitKey(1)
        if k == 27:
            bw = image[120:360,160:480]
            rt, bw = cv2.threshold(bw, 50,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            bw = cv2.bitwise_not(bw)
            cv2.imwrite("image.png", bw)
            img = load_img(img_path,grayscale = True, target_size=(28,28))
            x = img_to_array(img)
            x /= 255
            x = np.expand_dims(x,axis=0)
            model = predict()
            features = model.predict_classes(x,batch_size = 32)
            if features == old:
                print('True')
                print('sucsess')
                move(features)
                features = 0
            old = features

            print(features)
        if k == 42:
            break

    cv2.destroyAllWindows()

