# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:33:58 2018

@author: R-076
"""

import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler
import os
from tqdm import tqdm
import sys
def data_gen():
    # TRAIN DATA
    def create_data_test():
        train_labels = []
        train_samples = []
        # TEST DATA
        test_samples=[]
        for cnt in tqdm(range(1,2000)):
            random_young = randint(13,64)
            train_samples.append(random_young)
            train_labels.append(0)
            random_old = randint(64,100)
            train_samples.append(random_old)
            train_labels.append(1)
        train_labels= np.array(train_labels)
        train_samples = np.array(train_samples)
        
        scalar = MinMaxScaler(feature_range=(0,1))
        scaled_train_sampels = scalar.fit_transform((train_samples).reshape(-1,1))
        np.save("train_labels",train_labels)
        np.save("scaled_train_sampels",scaled_train_sampels)
        return(scaled_train_sampels,train_labels)
    def load_data():
        if "train_labels.npy" in os.listdir(os.getcwd()) and "scaled_train_sampels.npy" in os.listdir(os.getcwd()):
            print("Found data files..")
            train_labels = np.load("train_labels.npy")
            scaled_train_sampels = np.load("scaled_train_sampels.npy")
            print("Train files loaded sucessfully !")
            return (scaled_train_sampels,train_labels)
        else:
            print("Not found..\n Creating data..") 
            scaled_train_sampels,train_labels = create_data_test()
            print("Data generated and imported sucessfully !")
            np.save("train_labels",train_labels)
            np.save("scaled_train_sampels",scaled_train_sampels)

    return load_data()
    
    
