import numpy as np
import pandas as pd
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import os
path='rp_and_tp/test_point/'
dir_name_list=[x for x in os.listdir(path)]
x_test=[]
y_label=[]
for y in dir_name_list:
    npy_name_list=[x for x in os.listdir(path+y)]
    print(y)
    print(len(npy_name_list))
    for z in npy_name_list:
        load = np.load(path+y+'/'+z)
        #print(np.size(load))
        y_label.append(''.join([k for k in y if k.isdigit()]))
        x_test.append(load.reshape(-1,))
        #print(str(path+y+z))
N = len(y_label)
print('N: ',N)
x_n_test = np.array(x_test).reshape(N,-1) # N, 200
y_n_label = np.array(y_label).reshape(N,-1) # N,1

feature_num = np.size(x_n_test,1)


model = keras.models.load_model('localization.h3')
comp = model.predict(  x_n_test ) 
ans = np.array([])
print(comp)
for i in range(comp.shape[0]):
    #datum = comp[i]
    #print('index: %d' % i)
    #print('encoded datum: %s' % datum)
    #decoded_datum = np.argmax(comp[i])+1
    decoded_datum=comp[i]
    print(decoded_datum)
    if ans.shape[0] == 0:
       ans = np.array([decoded_datum])
    else:
       ans = np.append(ans, decoded_datum)
#print(np.round(ans).reshape(-1,)[:20])
print((ans).reshape(-1,)[:20])
print(y_n_label.reshape(-1,)[:20])
out = pd.DataFrame([ans.reshape(-1,),y_n_label.reshape(-1,)]).T
out.columns = ["ans","label"]
out.to_csv('output_predict2.csv',index = False) # to_csv : 輸出檔案
     
