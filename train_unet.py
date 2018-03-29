import cv2
import numpy as np
import os
import keras
import random


from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Reshape
from keras.optimizers import Adam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class UNET(object):
    def __init__(self):
        self.img_size = (512,512)
        self.train_image_path = "./data/train/image/"
        self.train_label_path = "./data/train/label/"
        self.test_image_path = "./data/test/image/"
        self.train_batch_size = 1
        self.val_batch_size = 1
        self.val_size =0.2
        self.learningrate = 1e-3
        self.epoch = 240

        self.model = self.Network()


    def Network(self):
        inputs = Input(shape=(self.img_size[0], self.img_size[1], 1), batch_shape=None)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(input=inputs, output=conv10)

        return model

    def train_network(self):
        images_label_list_train, images_label_list_val = self.load_images()
        self.model.compile(optimizer=Adam(lr=self.learningrate), loss='binary_crossentropy', metrics=['accuracy'])

        print("begin")
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath='unet_epoch{epoch:02d}_valacc{val_acc:.2f}_valloss{val_loss:.2f}.hdf5', monitor='val_acc',
            verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        reducelr=keras.callbacks.ReduceLROnPlateau(monitor='train_loss', factor=0.1, patience=4, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
        self.model.fit_generator(self.generate_data(images_label_list_train), \
                                 samples_per_epoch=self.train_batch_size, \
                                 nb_epoch=self.epoch, \
                                 validation_data=self.generate_data(images_label_list_val), \
                                 nb_val_samples=self.val_batch_size, \
                                 verbose=1, \
                                 nb_worker=1, \
                                 callbacks=[checkpointer,reducelr])


    def load_images(self):
        images_label_list_train = []
        images_label_list_val = []
        picnames = os.listdir(self.train_image_path)
        for i in range(0, len(picnames)):
                if i % int(len(picnames) / (len(picnames) * self.val_size)) != 0:
                    images_label_list_train.append([self.train_image_path + picnames[i], self.train_label_path + picnames[i]])
                else:
                    images_label_list_val.append([self.train_image_path + picnames[i], self.train_label_path + picnames[i]])

        return (images_label_list_train, images_label_list_val)

    def generate_data(self, images_label_list_train):
        while True:
            random.shuffle(images_label_list_train)
            X_image = []
            Y_label = []
            count = 0
            for image_label in images_label_list_train:
                img = cv2.imread(image_label[0], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0
                X_image.append(img)

                label = cv2.imread(image_label[1], cv2.IMREAD_GRAYSCALE)
                label = cv2.resize(label, self.img_size)
                label = label.astype('float32') / 255.0
                label[label > 0.5] = 1
                label[label <= 0.5] = 0
                Y_label.append(label)
                count += 1
                if count == self.train_batch_size:
                    count = 0
                    out_image=np.asarray(X_image).reshape(self.train_batch_size,self.img_size[0],self.img_size[1],1)
                    out_label=np.asarray(Y_label).reshape(self.train_batch_size,self.img_size[0],self.img_size[1],1)
                    yield (out_image,out_label)
                    X_image = []
                    Y_label = []



if __name__ == '__main__':
    unet = UNET()
    unet.train_network()
