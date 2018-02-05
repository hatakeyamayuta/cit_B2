import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from  keras.preprocessing.image import img_to_array
from  keras.preprocessing.image import load_img
model=VGG16(weights='imagenet',include_top=True)
img_path='image.png'
if __name__=="__main__":
    capture = cv2.VideoCapture(0)
    
    if capture.isOpened() is False:
        print("IO Error")

    cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

    while True:

        ret, image = capture.read()

        if ret == False:
            continue

        cv2.imshow("Capture", image)
       
        k = cv2.waitKey(1)
        if k == 27:
            cv2.imwrite("image.png", image)
            img = load_img(img_path, target_size=(224,224))
            x=img_to_array(img)
            x=np.expand_dims(x,axis=0)
            print(x.shape)
            x=preprocess_input(x)
            model.summary()
            features= model.predict(x)

            from keras.applications.vgg16 import decode_predictions
            results= decode_predictions(features, top=5)[0]
            for result in results:
                print(result)
        if k == 42:
            break

    cv2.destroyAllWindows()

