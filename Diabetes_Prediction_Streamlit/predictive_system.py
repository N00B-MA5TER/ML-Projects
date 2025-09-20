# -*- coding: utf-8 -*-

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open(r"C:\Users\SUBHADIP\Downloads\trained_model.sav", 'rb')) 

input_data = (5,166,72,19,175,25.8,0.587,51)

#changing the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)

if (prediction[0] == 0):
    print("The patient is Non-diabetic")
else:
    print("The patient is a Diabetic")