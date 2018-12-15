# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 12:18:35 2018

@author: R-076
"""



import os
from random import randint
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class Testing_Data():
    test_sampels=[]
    test_labels=[]

    def __init__(self):
        pass
    def test_data_generator(self):
        # Generating 100 sampels
        for i in range(100):
            # old
            self.test_sampels.append(randint(64,100))
            self.test_labels.append(1)
            
            # young
            self.test_sampels.append(randint(1,64))
            self.test_labels.append(0)
        
        test_sampels = np.array(self.test_sampels)
        test_labels = np.array(self.test_labels)
        
        scalar = MinMaxScaler(feature_range=(0,1))
        test_sampels = scalar.fit_transform((test_sampels).reshape(-1,1))
       
        # np.save("test_sampels",test_sampels)
        # np.save("test_labels",test_labels)
        
        return test_sampels
#obj = Testing_Data()
#print(obj.test_data_generator())