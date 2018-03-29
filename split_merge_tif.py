'''
split 30 single images from an array of images : train-volume.tif label-volume.tif test-volume.tif
'''
from libtiff import TIFF3D,TIFF
import os

input_tif = ("./ISBI/train-volume.tif","./ISBI/train-labels.tif","./ISBI/test-volume.tif")
output_dir=("./data/train/image/","./data/train/label/","./data/test/image/")
rewrite_tif = ("./data/train-volume.tif","./data/train-labels.tif","./data/test-volume.tif")


def split_img():
    '''
    split a tif volume into single tif
    '''
    for num,t in enumerate(input_tif):
        imgdir = TIFF3D.open(t)
        imgarr = imgdir.read_image()
        for i in range(imgarr.shape[0]):
            imgname = output_dir[num] + ("%03d.tif"%(i))
            img = TIFF.open(imgname,'w')
            img.write_image(imgarr[i])

def merge_img():
    '''
    merge single tif into a tif volume
    '''
    for num,tif_name in enumerate(rewrite_tif):
        imgdir = TIFF3D.open(tif_name,'w')
        imgarr = []
        tif_list=os.listdir(output_dir[num])
        for i in range(len(tif_list)):
            img = TIFF.open(output_dir[num] + tif_list[i])
            imgarr.append(img.read_image())
        imgdir.write_image(imgarr)

if __name__ == "__main__":

    #merge_img()
    split_img()



