# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:12:55 2018

@author: R-076
"""
import data_generator
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.models import load_model
import test_set
from pprint import pprint
import os
import sys

# Training data
scaled_train_sampels,train_labels = data_generator.data_gen()

# Testing data
obj = test_set.Testing_Data()
test_sampels = obj.test_data_generator()

def _neural_net_config():
    # NEURAL NET
    model = Sequential([])
    model.add(Dense(16,input_shape =(1,),activation="relu"))
    model.add(Dense(32,activation="relu"))
    model.add(Dense(2,activation="softmax"))
    #model.summary()

    model.compile(Adam(lr=.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(scaled_train_sampels,train_labels,batch_size=10,epochs = 20,shuffle = True, verbose = 2)
    model.save("medical_prediction.h5")
    return model
cwd = os.getcwd()
if 'medical_prediction.h5' in os.listdir(cwd):
    print("Model found !\nLoading Model")
    model = load_model("medical_prediction.h5")
    print("Model loaded !")
else:
    print("Couldn\'t find the model.. \n")
    model = _neural_net_config()
    
    

model.summary()
#predictions = model.predict(test_sampels,batch_size = 10, verbose = 2)
