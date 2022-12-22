import os.path
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras.applications.mobilenet_v3 as mobilenetv3
import cv2
from keras.models import model_from_json

class modelThread():
    IMAGE_WIDTH = 0
    IMAGE_HEIGHT = None
    classes = None
    model = None
    def __int__(self,path):
        super().__init__()
        self.IMAGE_WIDTH = 224
        self.IMAGE_HEIGHT = 224
        self.classes = {0: 'bag', 1: 'battery', 2: 'biological', 3: 'brown-glass', 4: 'cardboard', 5: 'clothes', 6: 'green-glass', 7: 'metal', 8: 'paper', 9: 'plastic', 10: 'shoes', 11: 'trash', 12: 'white-glass'}
        json_file = open(path + 'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(path + "model.h5")
        print(self.model.summary)


    def predict(self,image):
        img = cv2.resize(image,(224 ,224))
        index_list= self.model.predict(np.expand_dims(img, axis=0))
        index = np.argmax(index_list)
        if index == 0 or index == 11:
            text_shape = cv2.getTextSize("Red Lid Bin", cv2.FONT_HERSHEY_PLAIN, 2, 2)
            cv2.putText(image, "Red Lid Bin",((image.shape[0]//2)-(text_shape[0][0]//2) - 20, (image.shape[1]//2) - (text_shape[0][1]//2) +50), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255), 2)
            return image,index
        elif index ==3 or   index == 4 or index ==6 or index ==7 or index ==9 or index ==12 or index ==8 :
            text_shape = cv2.getTextSize("Yellow Lid Bin", cv2.FONT_HERSHEY_PLAIN, 2, 2)
            cv2.putText(image, "Yellow Lid Bin", ((image.shape[0]//2)-(text_shape[0][0]//2)- 20, (image.shape[1]//2) - (text_shape[0][1]//2)+50)  , cv2.FONT_HERSHEY_PLAIN, 2, (255,255, 0), 2)
            return image,index
        elif index == 2 :
            text_shape = cv2.getTextSize("Green Lid Bin", cv2.FONT_HERSHEY_PLAIN, 2, 2)
            cv2.putText(image, "Green Lid Bin", ((image.shape[0]//2)-(text_shape[0][0]//2)- 20, (image.shape[1]//2) - (text_shape[0][1]//2)  +50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            return image,index
        elif index == 1:
            text_shape = cv2.getTextSize("To Find E-waste Location", cv2.FONT_HERSHEY_PLAIN, 2, 2)
            cv2.putText(image, "To Find E-waste Location", ((image.shape[0]//2)-(text_shape[0][0]//2) - 20, (image.shape[1]//2) - (text_shape[0][1]//2)  +50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
            return image,index
        elif  index ==5 or index ==10:
            text_shape = cv2.getTextSize("To Find Clothes Location", cv2.FONT_HERSHEY_PLAIN, 2, 2)
            cv2.putText(image, "To Find Clothes Location", ((image.shape[0]//2)-(text_shape[0][0]//2)- 20, (image.shape[1]//2) - (text_shape[0][1]//2) +50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            return image,index
    #
    def run(self):
        print(os.getcwd())

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     img = cv2.imread("./garbage_classification/bag/bag1.jpg")
#     model = modelThread()
#     model.__int__()
#     result = model.predict(img)
#     cv2.imshow("image", result)
#     cv2.waitKey(0)