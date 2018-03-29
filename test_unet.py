import time
import os
import cv2
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class UNET(object):
    def __init__(self):
        self.model_name='unet_epoch215_valacc0.91_valloss0.20_1.hdf5'
        self.input_shape=(512,512)
        self.model=load_model(self.model_name)

    def pre_process_image(self,image_name):
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.input_shape)
        image = image / 255
        image = image.reshape(1,self.input_shape[0], self.input_shape[1],1)
        return image

    def infer(self,input_image):
        output_mask = self.model.predict(input_image, verbose=1)
        output_mask=output_mask.reshape(self.input_shape[0],self.input_shape[1],1)
        return output_mask


if __name__=="__main__":
    image_name="./data/test/image/000.tif"
    unet=UNET()
    image=unet.pre_process_image(image_name)
    start_time=time.time()
    output_mask=unet.infer(image)
    print("time:", (time.time() - start_time))
    cv2.imshow("result", output_mask)
    cv2.waitKey()



