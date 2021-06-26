# Imports

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os
import scipy.io
import glob
from tqdm import tqdm 
import random
import pickle
import cv2
import time
import scipy.optimize
from shapely.geometry import Polygon
import math
import traceback
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon
import tensorflow as tf
from Recognition_Pipeline import Recognition_Pipeline
import math
import pydot
import traceback
try:
    import queue
except ImportError:
    import Queue as queue
import warnings
warnings.filterwarnings("ignore")

# Prepare models

# Detector Model

class Deconv(tf.keras.layers.Layer):
    
    def __init__(self,name="deconv"):
        super().__init__(name)
        self.inp_shape = 0
        self.upsample = None
        self.conv = None
        self.bn = None
        
    def build(self,imshape):
        self.inp_shape=imshape
        self.upsample=tf.keras.layers.UpSampling2D(size=(2,2),interpolation='bilinear',data_format='channels_last',)
        self.conv=tf.keras.layers.Conv2D(filters=self.inp_shape[-1]//2,kernel_size=3,padding='same',activation='relu',
                                         kernel_initializer=tf.keras.initializers.GlorotNormal(seed=12),
                                         kernel_regularizer=tf.keras.regularizers.l2(1e-5), use_bias=False)
        self.bn=tf.keras.layers.BatchNormalization()
   
    def call(self,X):
    
        x=self.upsample(X)
        x=self.conv(x)
        x=self.bn(x)
        return x

resnet=tf.keras.applications.ResNet50(input_shape=(512,512,3),include_top=False,weights='imagenet')
tf.keras.backend.clear_session()

x = resnet.get_layer('conv5_block3_out').output

x = Deconv('Deconv1')(x)
x = tf.keras.layers.add([x,resnet.get_layer('conv4_block6_out').output])

x = Deconv('Deconv2')(x)
x = tf.keras.layers.add([x,resnet.get_layer('conv3_block4_out').output])

x = Deconv('Deconv3')(x)
x = tf.keras.layers.add([x,resnet.get_layer('conv2_block3_out').output])

# --- end ---

# text detector specific
x = Deconv('Deconv4')(x)
x = Deconv('Deconv5')(x)

score=tf.keras.layers.Conv2D(1,kernel_size=3,padding='same',activation='sigmoid')(x)

# Used this beacause sigmoid gives values in range of 0-1(as mentioned in git repository)
geo_map=tf.keras.layers.Conv2D(4,kernel_size=3,padding='same',activation='sigmoid')(x)*512

# Angles are assumed to be between [-45 to 45]
angle_map=(tf.keras.layers.Conv2D(1,kernel_size=3,padding='same',activation='sigmoid')(x)-0.5)*np.pi/2

out=tf.concat([score,geo_map,angle_map],axis=3)

detector=tf.keras.Model(resnet.input,out,name='detector')

# Load weights
detector.load_weights('detector_synth_CIDAR.h5')

# Recognition Model

# Text vocab for recognition
CHAR_VECTOR = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZÉ´-~`<>'.:;^/|!?$%#@&*()[]{}_+=,\\\""
NUM_CLASSES = len(CHAR_VECTOR)

# Recognition Model

ip = tf.keras.layers.Input(name ='input', shape = (64,128,3), dtype='float32')

x = tf.keras.layers.Conv2D(64, (3, 3), padding ='same', activation = 'relu', name ='conv1', kernel_initializer = 'he_normal')(ip) 
x = tf.keras.layers.Conv2D(64, (3, 3), padding ='same', activation = 'relu', name ='conv2', kernel_initializer = 'he_normal')(x) 
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 1), name = 'max-pool1')(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(128, (3, 3), padding ='same', activation = 'relu', name ='conv3', kernel_initializer = 'he_normal')(x) 
x = tf.keras.layers.Conv2D(128, (3, 3), padding ='same', activation = 'relu', name ='conv4', kernel_initializer = 'he_normal')(x) 
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 1), name = 'max-pool2')(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(256, (3, 3), padding ='same', activation = 'relu', name ='conv5', kernel_initializer = 'he_normal')(x) 
x = tf.keras.layers.Conv2D(256, (3, 3), padding ='same', activation = 'relu', name ='conv6', kernel_initializer = 'he_normal')(x) 
x = tf.keras.layers.MaxPooling2D(pool_size = (2, 1), name = 'max-pool3')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Reshape(target_shape= ((256,-1)), name='reshape')(x)

x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256,return_sequences=True))(x)
x = x=tf.keras.layers.Dense(NUM_CLASSES+2)(x)
out = tf.keras.activations.softmax(x)

recognizer=tf.keras.models.Model(ip, out)

# Load weights
recognizer.load_weights('recoginzer.h5')

# Use recognition pipeline
pipeline = Recognition_Pipeline()


# Streamlit Implementation

st.title('FOTS: Fast Oriented Text Spotting with a Unified Network')
st.header('Text detection and recognition')
st.write('We have text detection and recognition models which will give bounding box detection and recognize the text inside the given image.')

st.subheader('Below is a sample image')
image = Image.open('./ch4_training_images/img_43.jpg')
st.image(image, caption='Sample Image')

st.subheader('Upload a image to process')
uploaded_file = st.file_uploader('Select an image', type=['jpg', 'png'], accept_multiple_files=False, key=None, help=None)
if uploaded_file is not None:
    st.info('Uploaded file.')
    img_str = uploaded_file.read()
    img_data = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(img_data, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        im,txt=pipeline.recognize(img, detector, recognizer)
        if im is not None and txt is not None:
            st.success('Showing result...')
            st.image(im)
            st.write('Spotted text: {}'.format(','.join(txt)))
    except Exception as e:
        st.error('Some error occured. Try with another image.')
        
    

