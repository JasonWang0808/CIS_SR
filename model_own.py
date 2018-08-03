from utils_own import (
  read_data, 
  input_setup, 
  imsave,
  merge
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

try:
  xrange
except:
  xrange = range



class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size=35,
                 label_size=23,
                 batch_size=128,
                 c_dim=3,
                 checkpoint_dir=None,
                 sample_dir=None
                  ):
        self.sess = sess
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def load(self,checkpoint_dir):

         print("[！] Reading Checkpoints...")
         model_dir = "%s_%s" % ("srcnn",self.label_size)
         checkpoint_dir = os.path.join(checkpoint_dir,model_dir)
         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
         if ckpt and ckpt.model_checkpoint_path:
             ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
             self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))    
             return True
         else:
             return False



    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size,self.image_size,3 ])
        self.labels = tf.placeholder(tf.float32, [None, self.label_size,self.label_size,3 ])

        self.pred = self.model()

        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        #loss = tf.losses.softmax_cross_entropy(self.labels, self.pred)
        self.saver = tf.train.Saver()

    def model(self):
    	
        conv1 = tf.layers.conv2d(inputs=self.images,
                                  filters=64,
                                  kernel_size=[5, 5],
                                  padding="valid",
                                  activation=tf.nn.relu)
        conv1 = tf.layers.batch_normalization(conv1)
        conv2 = tf.layers.conv2d(inputs=conv1,
                                  filters=64,
                                  kernel_size=[3, 3],
                                  padding="valid",
                                  activation=tf.nn.relu)
        conv2 = tf.layers.batch_normalization(conv2)
        
       
        conv3 = tf.layers.conv2d(inputs=conv2,
                                  filters=3,
                                  kernel_size=[7, 7],
                                  padding="valid",
                                  activation=tf.nn.relu)
        print("++++++++++++")
        print(conv3.shape)
        return conv3
    
    def save(self, checkpoint_dir, step):
        model_name="SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name),
                    global_step=step)


    def train(self,config):
        if config.is_train:
            input_setup(self.sess,config)
        else:
            nx,ny = input_setup(self.sess,config)
       
        if config.is_train:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
        else:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

        train_data, train_label = read_data(data_dir)
      #  print(train_data.shape)
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)


        tf.initialize_all_variables().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
              print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.is_train:
              print("Training...")

              for ep in xrange(config.epoch):
                  batch_idxs = len(train_data) // config.batch_size
                  for idx in xrange(0,batch_idxs):
                      batch_images = train_data[idx*config.batch_size:(idx+1)*config.batch_size]
                      batch_labels = train_label[idx*config.batch_size:(idx+1)*config.batch_size]

                      counter += 1

                      _, err = self.sess.run([self.train_op,self.loss],feed_dict={self.images:batch_images,self.labels:batch_labels})

                      if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                              % ((ep+1), counter, time.time()-start_time, err))


                      if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)


        else:
            print("testing")

            result = self.pred.eval({self.images: train_data, self.labels: train_label})
            print("=============================")
            print(result.shape)

            result = merge(result,[nx,ny])
            print(result.shape)
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(),config.sample_dir)
            image_path = os.path.join(image_path,"test_image.png")
            imsave(result,image_path)





                    