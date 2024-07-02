# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:57:43 2021

@author: Grant
"""

import tensorflow 
import tensorflow_addons


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, MaxPooling2D, Conv2DTranspose, ReLU
from tensorflow.keras.layers import Add, Multiply, Subtract, Input, Conv1D, Conv2D, Conv3D, BatchNormalization, concatenate, Reshape, Lambda, UpSampling2D, UpSampling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import LeakyReLU

from tensorflow_addons.layers import InstanceNormalization

from tensorflow.keras.datasets import mnist

from tqdm import tqdm
import random
from PIL import Image

import os
import SimpleITK as sitk
#sick progress bar

import math



class GAN_art(object):
    
    def __init__(self,photo_dir,painting_dir,dest_dir,mom_photo_path):
        
        self.photo_dir = photo_dir
        self.painting_dir = painting_dir
        self.dest_dir = dest_dir
        self.mom_photo_path = mom_photo_path
        
        self.training_interrupted = False
        
        self.img_size = [256,256,3]  
        
        self.train_gen_paintings = []
        self.val_gen_paintings = []
        
        self.train_gen_photos = []
        self.val_gen_photos = []
        
        self.photos = self.get_photos()
        self.paintings = self.get_paintings()        
        
        self.partitioned_data_dict = self.partition_data()
        
        self.lr = 2e-4
        self.max_epoch_num = 100
        
        optimizer = Adam(self.lr)
        
        self.photo2painting_generator = self.build_generator()
        
        self.painting2photo_generator = self.build_generator()
        
        self.photo_discriminator = self.build_discriminator()
        
        self.photo_discriminator.compile(loss='mean_squared_error',
                           optimizer=optimizer,
                           metrics=['acc'])
        
        self.photo_discriminator.trainable = False
        
        self.painting_discriminator = self.build_discriminator()
        
        self.painting_discriminator.compile(loss='mean_squared_error',
                           optimizer=optimizer,
                           metrics=['acc'])
        
        self.painting_discriminator.trainable = False
        
        photo_input = Input(shape=(self.img_size[0],self.img_size[1],self.img_size[2]))
        painting_input = Input(shape=(self.img_size[0],self.img_size[1],self.img_size[2]))
        
        generated_painting = self.photo2painting_generator(photo_input)
        
        painting_pred = self.painting_discriminator(generated_painting)
        
        photo_recon = self.painting2photo_generator(generated_painting)
        
        photo_pred = self.photo_discriminator(photo_recon)
        
        painting_identity = self.photo2painting_generator(painting_input)
        
        self.cycleGAN = Model(inputs=[photo_input,painting_input],outputs=[photo_recon,painting_pred,photo_pred,painting_identity])
        
        self.cycleGAN.compile(loss=['mean_absolute_error','mean_squared_error','mean_squared_error','mean_absolute_error'],
                   optimizer=optimizer,
                   loss_weights=[10,1,1,1])
        
        self.mom_img_arr = self.get_mom_photo()
        
        
        
    def get_mom_photo(self):
        
        area = (400,400,656,656)
        image = Image.open(self.mom_photo_path)
        print(image.size)
        image = image.crop(area)           
        img_arr = np.array(image)
        
        img_arr = img_arr - 127.5 
        img_arr = img_arr / 127.5
        
        plt.imshow(img_arr)
        plt.axis('off')
        plt.show()
        
        return img_arr
        
        
    def get_photos(self):
        
        img_paths = []
        
        for dirName, subdirList, fileList in os.walk(self.photo_dir, topdown=True):
        
            for fname in fileList:
                
                img_paths.append(os.path.join(dirName,fname))
                
        arr_list = []
        
        for img_path in tqdm(img_paths,desc="Loading photos"):
            
            image = Image.open(img_path)            
            img_arr = np.array(image)
            
            img_arr = img_arr - 127.5 
            img_arr = img_arr / 127.5
            
            arr_list.append(img_arr)
            
        return arr_list
            
    def get_paintings(self):
        
        img_paths = []
        
        for dirName, subdirList, fileList in os.walk(self.painting_dir, topdown=True):
        
            for fname in fileList:
                
                img_paths.append(os.path.join(dirName,fname))
                
        arr_list = []
        
        for img_path in tqdm(img_paths,desc="Loading paintings"):
            
            image = Image.open(img_path)            
            img_arr = np.array(image)
            
            img_arr = img_arr - 127.5 
            img_arr = img_arr / 127.5 
            
            arr_list.append(img_arr)
            
        return arr_list
    
    def partition_data(self):
    
        train_photos = np.zeros(shape=(int(0.8*len(self.photos)),self.img_size[0],self.img_size[1],self.img_size[2]))
        val_photos = np.zeros(shape=(int(0.1*len(self.photos)),self.img_size[0],self.img_size[1],self.img_size[2]))
        test_photos = np.zeros(shape=(int(0.1*len(self.photos)),self.img_size[0],self.img_size[1],self.img_size[2]))

        train_paintings = np.zeros(shape=(int(0.8*len(self.paintings)),self.img_size[0],self.img_size[1],self.img_size[2]))
        val_paintings = np.zeros(shape=(int(0.1*len(self.paintings)),self.img_size[0],self.img_size[1],self.img_size[2]))
        test_paintings = np.zeros(shape=(int(0.1*len(self.paintings)),self.img_size[0],self.img_size[1],self.img_size[2]))       
        
        train_photo_count = 0
        val_photo_count = 0
        test_photo_count = 0
        
        train_painting_count = 0
        val_painting_count = 0
        test_painting_count = 0
        
        for i in range(len(self.photos)):
            
            if i < int(0.8*len(self.photos)):
                
                train_photos[train_photo_count,:,:,:] = self.photos[i]
                train_photo_count += 1
                
            if i > int(0.8*len(self.photos)) and i < int(0.9*len(self.photos)):
                
                val_photos[val_photo_count,:,:,:] = self.photos[i]                
                val_photo_count += 1
                
            if i > int(0.9*len(self.photos)):
                
                test_photos[test_photo_count,:,:,:] = self.photos[i]                
                test_photo_count += 1
                
                
        for i in range(len(self.paintings)):
            
            if i < int(0.8*len(self.paintings)):
                
                train_paintings[train_painting_count,:,:,:] = self.paintings[i]                
                train_painting_count += 1
                
            if i > int(0.8*len(self.paintings)) and i < int(0.9*len(self.paintings)):
                
                val_paintings[val_painting_count,:,:,:] = self.paintings[i]                
                val_painting_count += 1
                
            if i > int(0.9*len(self.paintings)):
                
                test_photos[test_painting_count,:,:,:] = self.paintings[i]                
                test_photo_count += 1
                
        return {"train_photos":train_photos,"val_photos":val_photos,"test_photos":test_photos,"train_paintings":train_paintings,"val_paintings":val_paintings,"test_paintings":test_paintings}
       
    def data_generator(self,batch_size,partition_type):
        
        photos = np.zeros(shape=(batch_size,self.img_size[0],self.img_size[1],self.img_size[2]))
        paintings = np.zeros(shape=(batch_size,self.img_size[0],self.img_size[1],self.img_size[2]))

        for i in range(batch_size):        
        
            if partition_type == "Training":
                
                photo_idx = np.random.randint(0,self.partitioned_data_dict["train_photos"].shape[0])
                painting_idx = np.random.randint(0,self.partitioned_data_dict["train_paintings"].shape[0])
                
                photos[i,:,:,:] = self.partitioned_data_dict["train_photos"][photo_idx]
                paintings[i,:,:,:] = self.partitioned_data_dict["train_paintings"][painting_idx]
            
            if partition_type == "Validation":
                
                photo_idx = np.random.randint(0,self.partitioned_data_dict["val_photos"].shape[0])
                painting_idx = np.random.randint(0,self.partitioned_data_dict["val_paintings"].shape[0])
                
                photos[i,:,:,:] = self.partitioned_data_dict["val_photos"][photo_idx]
                paintings[i,:,:,:] = self.partitioned_data_dict["val_paintings"][painting_idx]
                
            if partition_type == "Testing":
                
                photo_idx = np.random.randint(0,self.partitioned_data_dict["test_photos"].shape[0])
                painting_idx = np.random.randint(0,self.partitioned_data_dict["test_paintings"].shape[0])
                
                photos[i,:,:,:] = self.partitioned_data_dict["test_photos"][photo_idx]
                paintings[i,:,:,:] = self.partitioned_data_dict["test_paintings"][painting_idx]
                

        return photos,paintings
    
    def cycleGAN_data_generator(self,batch_size,partition_type):
        
        input_photos = np.zeros(shape=(batch_size,self.img_size[0],self.img_size[1],self.img_size[2]))
        input_paintings = np.zeros(shape=(batch_size,self.img_size[0],self.img_size[1],self.img_size[2]))
        
        recon_photos = np.zeros(shape=(batch_size,self.img_size[0],self.img_size[1],self.img_size[2]))
        painting_pred = np.ones(shape=(batch_size,30,30))
        photo_pred = np.ones(shape=(batch_size,30,30))        
        recon_paintings = np.zeros(shape=(batch_size,self.img_size[0],self.img_size[1],self.img_size[2]))
        
        for i in range(batch_size):        
        
            if partition_type == "Training":
                
                photo_idx = np.random.randint(0,self.partitioned_data_dict["train_photos"].shape[0])
                painting_idx = np.random.randint(0,self.partitioned_data_dict["train_paintings"].shape[0])
                
                input_photos[i,:,:,:] = self.partitioned_data_dict["train_photos"][photo_idx]
                input_paintings[i,:,:,:] = self.partitioned_data_dict["train_paintings"][painting_idx]
                
                recon_photos[i,:,:,:] = self.partitioned_data_dict["train_photos"][photo_idx]
                recon_paintings[i,:,:,:] = self.partitioned_data_dict["train_paintings"][painting_idx]
            
            if partition_type == "Validation":
                
                photo_idx = np.random.randint(0,self.partitioned_data_dict["val_photos"].shape[0])
                painting_idx = np.random.randint(0,self.partitioned_data_dict["val_paintings"].shape[0])
                
                input_photos[i,:,:,:] = self.partitioned_data_dict["val_photos"][photo_idx]
                input_paintings[i,:,:,:] = self.partitioned_data_dict["val_paintings"][painting_idx]
                
                recon_photos[i,:,:,:] = self.partitioned_data_dict["val_photos"][photo_idx]
                recon_paintings[i,:,:,:] = self.partitioned_data_dict["val_paintings"][painting_idx]
                
            if partition_type == "Testing":
                
                photo_idx = np.random.randint(0,self.partitioned_data_dict["test_photos"].shape[0])
                painting_idx = np.random.randint(0,self.partitioned_data_dict["test_paintings"].shape[0])
                
                input_photos[i,:,:,:] = self.partitioned_data_dict["test_photos"][photo_idx]
                input_paintings[i,:,:,:] = self.partitioned_data_dict["test_paintings"][painting_idx]
                
                recon_photos[i,:,:,:] = self.partitioned_data_dict["test_photos"][photo_idx]
                recon_paintings[i,:,:,:] = self.partitioned_data_dict["test_paintings"][painting_idx]
                

        return [input_photos,input_paintings],[recon_photos,painting_pred,photo_pred,recon_paintings]



    def selfAttention(self,img,kernels,dims):
        
        def softmax_lambda(x):
            return softmax(x,axis=1)
    
        def matmul(x):
            return K.batch_dot(x[0],x[1])
        
        conv1 = Conv2D(kernels,1)(img)
        reshape1 = Reshape((dims[0]*dims[1],kernels))(conv1)
        conv2 = Conv2D(kernels,1)(img)
        reshape2 = Reshape((kernels,dims[0]*dims[1]))(conv2)
        #
        matmul1 = Lambda(matmul)([reshape1,reshape2])
        softmax1 = Lambda(softmax_lambda)(matmul1)
        #
        conv3 = Conv2D(kernels,1)(img)
        reshape3 = Reshape((dims[0]*dims[1],kernels))(conv3)
        #
        matmul2 = Lambda(matmul)([softmax1,reshape3])
        reshape4 = Reshape((dims[0],dims[1],kernels))(matmul2)
        conv4 = Conv2D(kernels,1)(reshape4)
        #new = Add()([img,conv4])
        
        return conv4
    
    def res_block(self,input_FMs):
        
        conva = Conv2D(256,3,padding='same')(input_FMs)
        INa = InstanceNormalization()(conva)
        acta = ReLU()(INa)  
        
        convb = Conv2D(256,3,padding='same')(acta)
        INb = InstanceNormalization()(convb)
        FM_sum = Add()([INa,INb])
        
        return FM_sum
    
        
    def build_generator(self):
        
        photo = Input(shape=(self.img_size[0],self.img_size[1],self.img_size[2]))
        
        conv1 = Conv2D(64,7,padding='same')(photo)
        IN1 = InstanceNormalization()(conv1)
        act1 = ReLU()(IN1)
        
        conv2 = Conv2D(128,3,strides=(2,2),padding='same')(act1)
        IN2 = InstanceNormalization()(conv2)
        act2 = ReLU()(IN2)  
        
        conv3 = Conv2D(256,3,strides=(2,2),padding='same')(act2)
        IN3 = InstanceNormalization()(conv3)
        act3 = ReLU()(IN3)
        
        RM1 = self.res_block(act3)
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
        #painting = ReLU()(painting)
        
        model = Model(inputs=photo,outputs=painting)
    
        return model
    
    def build_discriminator(self):
        
        input_layer = Input(shape=(self.img_size[0],self.img_size[1],self.img_size[2]))
        
        conv1 = Conv2D(64,4,strides=(2,2),padding='same')(input_layer)
        IN1 = InstanceNormalization()(conv1)
        act1 = LeakyReLU()(IN1)
        
        conv2 = Conv2D(128,4,strides=(2,2),padding='same')(act1)
        IN2 = InstanceNormalization()(conv2)
        act2 = LeakyReLU()(IN2)
        
        conv3 = Conv2D(256,4,strides=(2,2),padding='same')(act2)
        IN3 = InstanceNormalization()(conv3)
        act3 = LeakyReLU()(IN3)
        
        conv4 = Conv2D(512,3,strides=(1,1),padding='valid')(act3)
        IN4 = InstanceNormalization()(conv4)
        act4 = LeakyReLU()(IN4)
        
        pred = Conv2D(1,3,activation='sigmoid',padding='same')(act4)
        
        model = Model(inputs=input_layer,outputs=pred)
        
        return model

    def train_GAN(self,epochs,batch_size): 
        
        train_total_losses = []
        val_total_losses = []
    
        train_recon_losses = []
        val_recon_losses = []
        
        train_painting_dis_losses = []
        val_painting_dis_losses = []
        
        train_photo_dis_losses = []
        val_photo_dis_losses = []
        
        train_identity_losses = []
        val_identity_losses = []
        
        train_painting_dis_losses = []
        val_painting_dis_losses = []
        
        
        train_photo_dis_losses = []
        val_photo_dis_losses = []
        
        
        train_total_losses_per_epoch = []
        val_total_losses_per_epoch = []
    
        train_recon_losses_per_epoch = []
        val_recon_losses_per_epoch = []
        
        train_painting_dis_losses_per_epoch = []
        val_painting_dis_losses_per_epoch = []
        
        train_photo_dis_losses_per_epoch = []
        val_photo_dis_losses_per_epoch = []
        
        train_identity_losses_per_epoch = []
        val_identity_losses_per_epoch = []
        
        train_painting_dis_losses_per_epoch = []
        val_painting_dis_losses_per_epoch = []
        
        
        train_photo_dis_losses_per_epoch = []
        val_photo_dis_losses_per_epoch = []
        
        
        steps_per_epoch = int(len(self.photos)/batch_size)
        

        dis_labels = np.zeros(shape=(2*batch_size,30,30))
        dis_labels[batch_size:,:,:] = np.ones(shape=(batch_size,30,30))
        
        look_back_num = 50
        
        if self.training_interrupted:
            
            #load and compile models
            
            optimizer = Adam(self.lr)
            
            self.photo2painting_generator = load_model(os.path.join(self.dest_dir,'MonetGAN_photo2painting_gen_current.h5'))
            
            self.painting2photo_generator = load_model(os.path.join(self.dest_dir,'MonetGAN_painting2photo_gen_current.h5'))
            
            self.photo_discriminator = load_model(os.path.join(self.dest_dir,'MonetGAN_photo_dis_current.h5'))
            
            self.photo_discriminator.compile(loss='mean_squared_error',
                               optimizer=optimizer,
                               metrics=['acc'])
            
            self.photo_discriminator.trainable = False
            
            self.painting_discriminator = load_model(os.path.join(self.dest_dir,'MonetGAN_painting_dis_current.h5'))
            
            self.painting_discriminator.compile(loss='mean_squared_error',
                               optimizer=optimizer,
                               metrics=['acc'])
            
            self.painting_discriminator.trainable = False
            
            photo_input = Input(shape=(self.img_size[0],self.img_size[1],self.img_size[2]))
            painting_input = Input(shape=(self.img_size[0],self.img_size[1],self.img_size[2]))
            
            generated_painting = self.photo2painting_generator(photo_input)
            
            painting_pred = self.painting_discriminator(generated_painting)
            
            photo_recon = self.painting2photo_generator(generated_painting)
            
            photo_pred = self.photo_discriminator(photo_recon)
            
            painting_identity = self.photo2painting_generator(painting_input)
            
            self.cycleGAN = Model(inputs=[photo_input,painting_input],outputs=[photo_recon,painting_pred,photo_pred,painting_identity])
            
            self.cycleGAN.compile(loss=['mean_absolute_error','mean_squared_error','mean_squared_error','mean_absolute_error'],
                       optimizer=optimizer,
                       loss_weights=[10,1,1,1])

            
            #load data  
            
            train_total_losses = list(np.loadtxt(os.path.join(self.dest_dir,"train_total_loss")))
            val_total_losses = list(np.loadtxt(os.path.join(self.dest_dir,"val_total_loss")))
            
            train_recon_losses = list(np.loadtxt(os.path.join(self.dest_dir,"train_recon_loss")))
            val_recon_losses = list(np.loadtxt(os.path.join(self.dest_dir,"val_recon_loss")))
            
            train_identity_losses = list(np.loadtxt(os.path.join(self.dest_dir,"train_identity_loss")))
            val_identity_losses = list(np.loadtxt(os.path.join(self.dest_dir,"val_identity_loss")))
            
            
            train_photo_dis_losses = list(np.loadtxt(os.path.join(self.dest_dir,"train_photo_dis_loss")))
            val_photo_dis_losses = list(np.loadtxt(os.path.join(self.dest_dir,"val_photo_dis_loss")))
            
            train_painting_dis_losses = list(np.loadtxt(os.path.join(self.dest_dir,"train_painting_dis_loss")))
            val_painting_dis_losses = list(np.loadtxt(os.path.join(self.dest_dir,"val_painting_dis_loss")))
        
        
        
        
        
        for epoch in range(epochs):
            for step in tqdm(range(steps_per_epoch),desc='Training Monet GAN. Epoch {}'.format(epoch+1)):
                
                train_input,train_output = self.cycleGAN_data_generator(batch_size,"Training")
                val_input,val_output = self.cycleGAN_data_generator(batch_size,"Validation")
                test_input,test_output = self.cycleGAN_data_generator(batch_size,"Validation")
        
                train_painting_pred = self.photo2painting_generator.predict(train_input[0])
                val_painting_pred = self.photo2painting_generator.predict(val_input[0])
                #test_painting_pred = self.photo2painting_generator.predict(test_input[0])
                
                train_photo_pred = self.painting2photo_generator.predict(train_input[1])
                val_photo_pred = self.painting2photo_generator.predict(val_input[1])
                #test_photo_pred = self.painting2photo_generator.predict(test_input[1])

                self.train_gen_paintings.append(train_painting_pred[0,:,:,:])
                self.val_gen_paintings.append(val_painting_pred[0,:,:,:])
                
                self.train_gen_photos.append(train_photo_pred[0,:,:,:])
                self.val_gen_photos.append(val_photo_pred[0,:,:,:])
                
                if len(self.train_gen_paintings) >= look_back_num:
                    
                    self.train_gen_paintings = self.train_gen_paintings[:look_back_num]
                    self.val_gen_paintings = self.val_gen_paintings[:look_back_num]
                    self.train_gen_photos = self.train_gen_photos[:look_back_num]
                    self.val_gen_photos = self.val_gen_photos[:look_back_num]
                
                    train_painting_idx = np.random.randint(look_back_num)
                    val_painting_idx = np.random.randint(look_back_num)
                    
                    train_photo_idx = np.random.randint(look_back_num)
                    val_photo_idx = np.random.randint(look_back_num)  
                    
                    train_painting_pred = np.expand_dims(self.train_gen_paintings[train_painting_idx],axis=0)
                    train_photo_pred = np.expand_dims(self.train_gen_photos[train_photo_idx],axis=0)
                    
                    val_painting_pred = np.expand_dims(self.train_gen_paintings[val_painting_idx],axis=0)
                    val_photo_pred = np.expand_dims(self.train_gen_photos[val_photo_idx],axis=0)
                                    
   
                    
                train_painting_dis_in = np.zeros(shape=(2*batch_size,train_input[1].shape[1],train_input[1].shape[2],train_input[1].shape[3]))
                val_painting_dis_in = np.zeros(shape=(2*batch_size,val_input[1].shape[1],val_input[1].shape[2],val_input[1].shape[3]))
                
                train_painting_dis_in[:batch_size,:,:,:] = train_painting_pred
                val_painting_dis_in[:batch_size,:,:,:] = val_painting_pred
                
                train_painting_dis_in[batch_size:,:,:,:] = train_output[0]
                val_painting_dis_in[batch_size:,:,:,:] = val_output[0]
                
                
                dis_painting_train_loss_list = self.painting_discriminator.train_on_batch(train_painting_dis_in,dis_labels)
                dis_painting_val_loss_list = self.painting_discriminator.test_on_batch(val_painting_dis_in,dis_labels)
         
                train_painting_dis_loss = dis_painting_train_loss_list[0]
                val_painting_dis_loss = dis_painting_val_loss_list[0]
                
                train_painting_dis_losses.append(train_painting_dis_loss)
                val_painting_dis_losses.append(val_painting_dis_loss)

                
                train_photo_dis_in = np.zeros(shape=(2*batch_size,train_output[3].shape[1],train_output[3].shape[2],train_output[3].shape[3]))
                val_photo_dis_in = np.zeros(shape=(2*batch_size,val_output[3].shape[1],val_output[3].shape[2],val_output[3].shape[3]))
                
                train_photo_dis_in[:batch_size,:,:,:] = train_photo_pred
                val_photo_dis_in[:batch_size,:,:,:] = val_photo_pred
                
                train_photo_dis_in[batch_size:,:,:,:] = train_output[3]
                val_photo_dis_in[batch_size:,:,:,:] = val_output[3]
                
                dis_photo_train_loss_list = self.photo_discriminator.train_on_batch(train_photo_dis_in,dis_labels)
                dis_photo_val_loss_list = self.photo_discriminator.test_on_batch(val_photo_dis_in,dis_labels)
         
                train_photo_dis_loss = dis_photo_train_loss_list[0]
                val_photo_dis_loss = dis_photo_val_loss_list[0]
                
                
                train_photo_dis_losses.append(train_photo_dis_loss)
                val_photo_dis_losses.append(val_photo_dis_loss)
                           
 
                
                gen_train_loss_list = self.cycleGAN.train_on_batch(train_input,train_output)
                gen_val_loss_list = self.cycleGAN.test_on_batch(val_input,val_output)
                
                train_total_loss = gen_train_loss_list[0]
                val_total_loss = gen_val_loss_list[0]
                
                train_recon_loss = gen_train_loss_list[1]
                val_recon_loss = gen_val_loss_list[1]
                
                train_painting_dis_loss = gen_train_loss_list[2]
                val_painting_dis_loss = gen_val_loss_list[2]
                
                train_photo_dis_loss = gen_train_loss_list[3]
                val_photo_dis_loss = gen_val_loss_list[3]
                
                train_identity_loss = gen_train_loss_list[4]
                val_identity_loss = gen_val_loss_list[4]
                
                train_total_losses.append(train_total_loss)
                val_total_losses.append(val_total_loss) 
                
                train_recon_losses.append(train_recon_loss)
                val_recon_losses.append(val_recon_loss) 
                
                train_painting_dis_losses.append(train_painting_dis_loss)
                val_painting_dis_losses.append(val_painting_dis_loss) 
                
                train_photo_dis_losses.append(train_photo_dis_loss)
                val_photo_dis_losses.append(val_photo_dis_loss) 
                
                train_identity_losses.append(train_identity_loss)
                val_identity_losses.append(val_identity_loss)
                
                np.savetxt(os.path.join(self.dest_dir,"train_total_loss"),train_total_losses)
                np.savetxt(os.path.join(self.dest_dir,"val_total_loss"),val_total_losses)
                
                np.savetxt(os.path.join(self.dest_dir,"train_recon_loss"),train_recon_losses)
                np.savetxt(os.path.join(self.dest_dir,"val_recon_loss"),val_recon_losses)
                
                np.savetxt(os.path.join(self.dest_dir,"train_identity_loss"),train_identity_losses)
                np.savetxt(os.path.join(self.dest_dir,"val_identity_loss"),val_identity_losses)
                
                np.savetxt(os.path.join(self.dest_dir,"train_photo_dis_loss"),train_photo_dis_losses)
                np.savetxt(os.path.join(self.dest_dir,"val_photo_dis_loss"),val_photo_dis_losses)
                
                np.savetxt(os.path.join(self.dest_dir,"train_painting_dis_loss"),train_painting_dis_losses)
                np.savetxt(os.path.join(self.dest_dir,"val_painting_dis_loss"),val_painting_dis_losses)


                
                
                if step == 0:
                
                    train_total_losses_per_epoch.append(train_total_loss)
                    val_total_losses_per_epoch.append(val_total_loss) 
                    
                    train_recon_losses_per_epoch.append(train_recon_loss)
                    val_recon_losses_per_epoch.append(val_recon_loss) 
                    
                    train_painting_dis_losses_per_epoch.append(train_painting_dis_loss)
                    val_painting_dis_losses_per_epoch.append(val_painting_dis_loss) 
                    
                    train_photo_dis_losses_per_epoch.append(train_photo_dis_loss)
                    val_photo_dis_losses_per_epoch.append(val_photo_dis_loss) 
                    
                    train_identity_losses_per_epoch.append(train_identity_loss)
                    val_identity_losses_per_epoch.append(val_identity_loss) 
                    
                    train_painting_dis_losses_per_epoch.append(train_painting_dis_losses[-1])
                    val_painting_dis_losses_per_epoch.append(val_painting_dis_losses[-1])
                    
                    train_photo_dis_losses_per_epoch.append(train_photo_dis_losses[-1])
                    val_photo_dis_losses_per_epoch.append(val_photo_dis_losses[-1])
                    

                
                if step > 0 or epoch > 0:
                    
                    if val_total_losses[-1] == np.min(val_total_losses):
                        self.photo2painting_generator.save(os.path.join(self.dest_dir,'MonetGAN_photo2painting_gen_best_val_loss.h5'))

                self.photo2painting_generator.save(os.path.join(self.dest_dir,'MonetGAN_photo2painting_gen_current.h5'))
                self.painting2photo_generator.save(os.path.join(self.dest_dir,'MonetGAN_painting2photo_gen_current.h5'))
                self.painting_discriminator.save(os.path.join(self.dest_dir,'MonetGAN_painting_dis_current.h5'))
                self.photo_discriminator.save(os.path.join(self.dest_dir,'MonetGAN_photo_dis_current.h5'))
            
            
            #Insert learning rate decay code that save and loads weights and recompiles the GAN models
            if epoch > self.max_epoch_num:
                
                
                models = [self.photo2painting_generator,self.painting2photo_generator,self.painting_discriminator,self.photo_discriminator]
                weights = [model.get_weights() for model in models]

                scale = self.lr/self.max_epoch_num

                self.lr *= scale               
                
                optimizer = Adam(self.lr)
        
                for m,model in enumerate(models):
                    
                    model.set_weights(weights[m])
                
                self.photo_discriminator.compile(loss='mean_squared_error',
                                   optimizer=optimizer,
                                   metrics=['acc'])
                
                self.photo_discriminator.trainable = False
                
                self.painting_discriminator.compile(loss='mean_squared_error',
                                   optimizer=optimizer,
                                   metrics=['acc'])
                
                self.painting_discriminator.trainable = False
                
                photo_input = Input(shape=(self.img_size[0],self.img_size[1],self.img_size[2]))
                painting_input = Input(shape=(self.img_size[0],self.img_size[1],self.img_size[2]))
                
                generated_painting = self.photo2painting_generator(photo_input)
                
                painting_pred = self.painting_discriminator(generated_painting)
                
                photo_recon = self.painting2photo_generator(generated_painting)
                
                photo_pred = self.photo_discriminator(photo_recon)
                
                painting_identity = self.photo2painting_generator(painting_input)
                
                self.cycleGAN = Model(inputs=[photo_input,painting_input],outputs=[photo_recon,painting_pred,photo_pred,painting_identity])
                
                self.cycleGAN.compile(loss=['mean_absolute_error','mean_squared_error','mean_squared_error','mean_absolute_error'],
                           optimizer=optimizer,
                           loss_weights=[10,1,1,1])
                 
            mom_painting = self.photo2painting_generator(np.expand_dims(self.mom_img_arr,axis=0))[0,:,:,:]
            
            plt.imshow(self.mom_img_arr)
            plt.title('Real Photo')
            plt.axis('off')
            plt.show()
            
            plt.imshow(test_input[1][0,:,:,:])
            plt.title('Real Painting')
            plt.axis('off')
            plt.show()
            
            plt.imshow(mom_painting)
            plt.title('Painting from Photo using Monet GAN')
            plt.axis('off')
            plt.show()
            
                
        plt.figure(figsize = (8,6))
        plt.title('Training and validation total loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        ta, = plt.plot(range(len(train_total_losses_per_epoch)), train_total_losses_per_epoch)
        va, = plt.plot(range(len(val_total_losses_per_epoch)), val_total_losses_per_epoch)
        plt.legend([ta, va], ['Training', 'Validation'])
        plt.savefig(os.path.join(self.dest_dir,'total loss cuves'))
        plt.show()
        
        plt.figure(figsize = (8,6))
        plt.title('Training and validation recon loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        ta, = plt.plot(range(len(train_recon_losses_per_epoch)), train_recon_losses_per_epoch)
        va, = plt.plot(range(len(val_recon_losses_per_epoch)), val_recon_losses_per_epoch)
        plt.legend([ta, va], ['Training', 'Validation'])
        plt.savefig(os.path.join(self.dest_dir,'recon loss cuves'))
        plt.show()
        
        plt.figure(figsize = (8,6))
        plt.title('Training and validation identity loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        ta, = plt.plot(range(len(train_identity_losses_per_epoch)), train_identity_losses_per_epoch)
        va, = plt.plot(range(len(val_identity_losses_per_epoch)), val_identity_losses_per_epoch)
        plt.legend([ta, va], ['Training', 'Validation'])
        plt.savefig(os.path.join(self.dest_dir,'identity loss cuves'))
        plt.show()
        
        plt.figure(figsize = (8,6))
        plt.title('Training and validation painting discriminator loss')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        ta, = plt.plot(range(len(train_painting_dis_losses_per_epoch)), train_painting_dis_losses_per_epoch)
        va, = plt.plot(range(len(val_painting_dis_losses_per_epoch)), val_painting_dis_losses_per_epoch)
        plt.legend([ta, va], ['Training', 'Validation'])
        plt.savefig(os.path.join(self.dest_dir,'discriminator painting loss cuves'))
        plt.show()
        
        plt.figure(figsize = (8,6))
        plt.title('Training and validation photo discriminator loss')
        plt.xlabel('Steps')
        plt.ylabel('Accuracy')
        ta, = plt.plot(range(len(train_photo_dis_losses_per_epoch)), train_photo_dis_losses_per_epoch)
        va, = plt.plot(range(len(val_photo_dis_losses_per_epoch)), val_photo_dis_losses_per_epoch)
        plt.legend([ta, va], ['Training', 'Validation'])
        plt.savefig(os.path.join(self.dest_dir,'discriminator photo loss cuves'))
        plt.show()
        
    
    
    
    
photo_dir = ""
painting_dir = ""
dest_dir = ""

mom_photo_path = ""
        
        
GAN_Art = GAN_art(photo_dir,painting_dir,dest_dir,mom_photo_path)

batch_size = 1
#Do not increase for now
epochs = 200

GAN_Art.train_GAN(epochs,batch_size)





