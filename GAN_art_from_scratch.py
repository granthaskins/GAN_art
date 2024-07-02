# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 02:32:30 2020

@author: Grant
"""

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, ReLU
from tensorflow.keras.layers import Add, Multiply, Subtract, Input, Conv1D, Conv2D, Conv2DTranspose, Conv3D, BatchNormalization, concatenate, Reshape, Lambda, UpSampling2D, UpSampling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.activations import relu

from PIL import Image

import os

from tqdm import tqdm

#from tensorflow_addons.layers import InstanceNormalization


class GAN_art_scratch(object):
    
    def __init__(self,painting_dir,dest_dir):
        
        
        self.painting_dir = painting_dir
        self.dest_dir = dest_dir

        self.lr = 1e-4
        
        self.img_size = [256,256,3]
        
        
        self.paintings = self.get_paintings()
        self.partitioned_data_dict = self.partition_data()
        
        
        self.optimizer = Adam(self.lr)
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        self.discriminator.compile(loss='mean_squared_error',
                           optimizer=self.optimizer,
                           metrics=['acc'])
        
        self.discriminator.trainable = False

        
        GAN_inputs = Input(shape=(int(self.img_size[0]/4),int(self.img_size[1]/4),self.img_size[2]))
        generator_output = self.generator(GAN_inputs)
        
        pred = self.discriminator(generator_output)
        
        self.GAN = Model(inputs=GAN_inputs,outputs=pred)
        
        self.GAN.compile(loss=['mean_squared_error'],
                           optimizer=self.optimizer,
                           metrics=['acc'])

    def get_paintings(self):
        
        img_paths = []
        
        for dirName, subdirList, fileList in os.walk(self.painting_dir, topdown=True):
        
            for fname in fileList:
                
                img_paths.append(os.path.join(dirName,fname))
                
        arr_list = []
        
        for img_path in tqdm(img_paths,desc="Loading paintings"):
            
            image = Image.open(img_path)            
            img_arr = np.array(image)
            
            img_arr = img_arr / 255.
            
            # img_arr = img_arr - 127.5 
            # img_arr = img_arr / 127.5 
            
            arr_list.append(img_arr)
            
        return arr_list
    
    def partition_data(self):
    
        train_paintings = np.zeros(shape=(int(0.8*len(self.paintings)),self.img_size[0],self.img_size[1],self.img_size[2]))
        val_paintings = np.zeros(shape=(int(0.1*len(self.paintings)),self.img_size[0],self.img_size[1],self.img_size[2]))
        test_paintings = np.zeros(shape=(int(0.1*len(self.paintings)),self.img_size[0],self.img_size[1],self.img_size[2]))               
        
        train_painting_count = 0
        val_painting_count = 0
        test_painting_count = 0                
                
        for i in range(len(self.paintings)):
            
            if i < int(0.8*len(self.paintings)):
                
                train_paintings[train_painting_count,:,:,:] = self.paintings[i]                
                train_painting_count += 1
                
            if i > int(0.8*len(self.paintings)) and i < int(0.9*len(self.paintings)):
                
                val_paintings[val_painting_count,:,:,:] = self.paintings[i]                
                val_painting_count += 1
                
            if i > int(0.9*len(self.paintings)):
                
                test_paintings[test_painting_count,:,:,:] = self.paintings[i]                
                test_painting_count += 1
                
        return {"train_paintings":train_paintings,"val_paintings":val_paintings,"test_paintings":test_paintings}

       
    def data_generator(self,batch_size,partition_type):
        
        latents = np.zeros(shape=(batch_size,int(self.img_size[0]/4),int(self.img_size[1]/4),self.img_size[2]))
        paintings = np.zeros(shape=(batch_size,self.img_size[0],self.img_size[1],self.img_size[2]))

        for i in range(batch_size):   
            
            latents[i,:,:,:] = np.random.normal(loc=0.0, scale=1.0, size=(int(self.img_size[0]/4),int(self.img_size[1]/4),self.img_size[2]))
        
            if partition_type == "Training":
                
                painting_idx = np.random.randint(0,self.partitioned_data_dict["train_paintings"].shape[0])
                
                paintings[i,:,:,:] = self.partitioned_data_dict["train_paintings"][painting_idx]
            
            if partition_type == "Validation":
                
                painting_idx = np.random.randint(0,self.partitioned_data_dict["val_paintings"].shape[0])
                
                paintings[i,:,:,:] = self.partitioned_data_dict["val_paintings"][painting_idx]
                
            if partition_type == "Testing":
                
                painting_idx = np.random.randint(0,self.partitioned_data_dict["test_paintings"].shape[0])
                
                paintings[i,:,:,:] = self.partitioned_data_dict["test_paintings"][painting_idx]
                

        return latents,paintings

    def res_block(self,input_FMs):
        
        conva = Conv2D(256,3,padding='same')(input_FMs)
        BNa = BatchNormalization()(conva)
        #INa = InstanceNormalization()(conva)
        acta = ReLU()(BNa)  
        
        convb = Conv2D(256,3,padding='same')(acta)
        #INb = InstanceNormalization()(convb)
        BNb = BatchNormalization()(convb)
        FM_sum = Add()([BNa,BNb])
        
        return FM_sum

    def build_generator(self):
        
        latent = Input(shape=(int(self.img_size[0]/4),int(self.img_size[1]/4),self.img_size[2]))

        RM1 = self.res_block(latent)
        RM2 = self.res_block(RM1)
        RM3 = self.res_block(RM2)
        RM4 = self.res_block(RM3)
        RM5 = self.res_block(RM4)
        RM6 = self.res_block(RM5)
        RM7 = self.res_block(RM6)
        RM8 = self.res_block(RM7)
        RM9 = self.res_block(RM8)
        
        conv1T = Conv2DTranspose(128,3,strides=(2,2),padding='same')(RM9)
        conv2T = Conv2DTranspose(64,3,strides=(2,2),padding='same')(conv1T)
        
        painting = Conv2D(3,7,padding='same')(conv2T)
        painting = relu(painting,max_value=1.)
        
        model = Model(inputs=latent,outputs=painting)
    
        return model
    
    def build_discriminator(self):
        
        input_layer = Input(shape=(self.img_size[0],self.img_size[1],self.img_size[2]))
        
        conv1 = Conv2D(64,4,strides=(2,2),padding='same')(input_layer)
        BN1 = BatchNormalization()(conv1)
        #IN1 = InstanceNormalization()(conv1)
        act1 = LeakyReLU()(BN1)
        
        conv2 = Conv2D(128,4,strides=(2,2),padding='same')(act1)
        BN2 = BatchNormalization()(conv2)
        #IN2 = InstanceNormalization()(conv2)
        act2 = LeakyReLU()(BN2)
        
        conv3 = Conv2D(256,4,strides=(2,2),padding='same')(act2)
        BN3 = BatchNormalization()(conv3)
        #IN3 = InstanceNormalization()(conv3)
        act3 = LeakyReLU()(BN3)
        
        conv4 = Conv2D(512,3,strides=(1,1),padding='valid')(act3)
        BN4 = BatchNormalization()(conv4)
        #IN4 = InstanceNormalization()(conv4)
        act4 = LeakyReLU()(BN4)
        
        pred = Conv2D(1,3,activation='sigmoid',padding='same')(act4)
        
        model = Model(inputs=input_layer,outputs=pred)
        
        return model


    def train_GAN(self,epochs,batch_size):
    
        train_gen_losses = []
        val_gen_losses = []
        
        train_dis_losses = []
        val_dis_losses = []
        
        train_gen_losses_per_epoch = []
        val_gen_losses_per_epoch = []
        
        train_dis_losses_per_epoch = []
        val_dis_losses_per_epoch = []
        
        steps_per_epoch = int(len(self.paintings)/batch_size)
        
        real_labels = np.ones(shape=(batch_size,30,30))
        
        dis_labels = np.zeros(shape=(2*batch_size,30,30))
        dis_labels[batch_size:,:,:] = np.ones(shape=(batch_size,30,30))
        
        for epoch in range(epochs):
            for step in tqdm(range(steps_per_epoch),desc="Training GAN; epoch {}".format(epoch+1)):
                
                train_input,train_output = self.data_generator(batch_size,"Training")
                val_input,val_output = self.data_generator(batch_size,"Validation")
        
                train_pred = self.generator.predict(train_input)
                val_pred = self.generator.predict(val_input)
                
                train_dis_in = np.zeros(shape=(2*batch_size,train_output.shape[1],train_output.shape[2],train_output.shape[3]))
                val_dis_in = np.zeros(shape=(2*batch_size,val_output.shape[1],val_output.shape[2],val_output.shape[3]))
                
                train_dis_in[:batch_size,:,:,:] = train_pred
                val_dis_in[:batch_size,:,:,:] = val_pred
                
                train_dis_in[batch_size:,:,:,:] = train_output
                val_dis_in[batch_size:,:,:,:] = val_output
                
                dis_train_loss_list = self.discriminator.train_on_batch(train_dis_in,dis_labels)
                dis_val_loss_list = self.discriminator.test_on_batch(val_dis_in,dis_labels)
         
                train_dis_loss = dis_train_loss_list[0]
                val_dis_loss = dis_val_loss_list[0]
                
                
                train_dis_losses.append(train_dis_loss)
                val_dis_losses.append(val_dis_loss)
                
                
                gen_train_loss_list = self.GAN.train_on_batch(train_input,real_labels)
                gen_val_loss_list = self.GAN.test_on_batch(val_input,real_labels)
                
                train_gen_loss = gen_train_loss_list[0]
                val_gen_loss = gen_val_loss_list[0]
                
                train_gen_losses.append(train_gen_loss)
                val_gen_losses.append(val_gen_loss)
        
                if step == 0:
                
                    train_gen_losses_per_epoch.append(train_gen_loss)
                    val_gen_losses_per_epoch.append(val_gen_loss) 
                    
                    train_dis_losses_per_epoch.append(train_dis_losses[-1])
                    val_dis_losses_per_epoch.append(val_dis_losses[-1])

                    
                
                if step > 0:
                    
                    if val_gen_losses[-1] == np.min(val_gen_losses):
                        self.generator.save(os.path.join(self.dest_dir,'Monet_GAN_scratch_val_loss.h5'))

                self.generator.save(os.path.join(self.dest_dir,'Monet_GAN_scratch_current.h5'))
                    
   
            
            plt.imshow(val_output[0,:,:,:])
            plt.title('Real Painting')
            plt.axis('off')
            plt.show()
            
            plt.imshow(val_pred[0,:,:,:])
            plt.title('Painting by GAN')
            plt.axis('off')
            plt.show()
            
                
        plt.figure(figsize = (8,6))
        plt.title('Training and validation generator loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        ta, = plt.plot(range(len(train_gen_losses_per_epoch)), train_gen_losses_per_epoch)
        va, = plt.plot(range(len(val_gen_losses_per_epoch)), val_gen_losses_per_epoch)
        plt.legend([ta, va], ['Training', 'Validation'])
        plt.savefig(os.path.join(self.dest_dir,'total loss cuves'))
        plt.show()
        
        plt.figure(figsize = (8,6))
        plt.title('Training and validation discriminator loss')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        ta, = plt.plot(range(len(train_dis_losses_per_epoch)), train_dis_losses_per_epoch)
        va, = plt.plot(range(len(val_dis_losses_per_epoch)), val_dis_losses_per_epoch)
        plt.legend([ta, va], ['Training', 'Validation'])
        plt.savefig(os.path.join(self.dest_dir,'discriminator loss cuves'))
        plt.show()
        
        



painting_dir ='/home/grant/Dropbox/PC/Downloads/gan-getting-started/monet_jpg'
dest_dir = '/Users/grant/Desktop/GAN_art_files'

        
        
GAN_Art = GAN_art_scratch(painting_dir,dest_dir)

batch_size = 16

epochs = 10

GAN_Art.train_GAN(epochs,batch_size)
        
        


