import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np
import cv2
import tensorflow as tf

try:
  xrange
except:
  xrange = range
  
FLAGS = tf.app.flags.FLAGS

'''
## 1. read_data
## 2. imsave
## 3. input_setup
    3.1 prepare_data (get address)
    3.2 imread 
    3.3 preprocess (return label_, input_)
    3.4 cut
    3.5 make_data (save)
'''

def read_data(path):

    with h5py.File(path,'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data,label

def imsave(image,path):
    return scipy.misc.imsave(path, image)

def prepare_data(sess,dataset):
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))

    return data

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def modcrop(image,scale=3):
    h,w,_ = image.shape
    h = h - np.mod(h,scale)
    w = w - np.mod(w,scale)
    image = image[0:h,0:w]
    return image

def preprocess(path,config):
    image = imread(path)
   # label_ = modcrop(image)
    label_ = image
    input_ = cv2.GaussianBlur(image,(5,5),3)
    label_ = label_ / 255.
    input_ = input_ / 255.
    image_path = os.path.join(os.getcwd(), config.sample_dir)
    if not config.is_train:
    	image_path = os.path.join(image_path, "test_begin_image.png")
    	imsave(input_,image_path)
 
    return input_, label_

def make_data(sess, data, label):



  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)


def input_setup(sess,config):
  if config.is_train:
    data = prepare_data(sess,dataset='Train')
  else:
    data = prepare_data(sess,dataset='Test')

  sub_input_sequence = []
  sub_label_sequence = []

  padding = abs(config.image_size - config.label_size) / 2 # 6

  if config.is_train:
    for i in xrange(len(data)):
      input_, label_ = preprocess(data[i],config)

      h,w, _ = input_.shape

      for x in range(0, h-config.image_size+1, config.stride):
        for y in range(0, w-config.image_size+1, config.stride):
          sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
          sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]

          # Make channel value
          sub_input = sub_input.reshape([config.image_size, config.image_size, 3])
          sub_label = sub_label.reshape([config.label_size, config.label_size, 3])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)


  else:
    input_,label_ = preprocess(data[2],config)

    h,w ,_ = input_.shape
    print("stride is ")
    print(config.stride)
    print("label_size is")
    print(config.label_size)

    nx = ny = 0
    for x in range(0, h-config.image_size+1, config.stride):
        nx += 1; ny = 0
        for y in range(0, w-config.image_size+1, config.stride):
            ny += 1
            sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
            sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]
            
            sub_input = sub_input.reshape([config.image_size, config.image_size, 3])
            sub_label = sub_label.reshape([config.label_size, config.label_size, 3])

            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 3]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 3]
  print(arrdata.shape)
  print(arrlabel.shape)
  make_data(sess, arrdata, arrlabel)

  if not config.is_train:
      print(nx)
      print(ny)
      return nx,ny



def merge(images, size):
    h,w = images.shape[1],images.shape[2]
    img = np.zeros((h*size[0],w*size[1],3))
    print(h)
    print(w)
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
        
    return img
