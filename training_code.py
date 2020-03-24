import numpy as np
import pandas as pd
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras.callbacks import EarlyStopping

import os
import matplotlib.pyplot as plt
path='rp_and_tp/reference_point/'
dir_name_list=[x for x in os.listdir(path)]
x_train=[]
y_label=[]
for y in dir_name_list:
    npy_name_list=[x for x in os.listdir(path+y)]
    print(y)
    print(len(npy_name_list))
    for z in npy_name_list:
        load = np.load(path+y+'/'+z)
        #print(np.size(load))
        y_label.append(''.join([k for k in y if k.isdigit()]))
        x_train.append(load.reshape(-1,))
        #print(str(path+y+z))
N = len(y_label)
print('N: ',N)
x_n_train = np.array(x_train).reshape(N,-1) # N, 100
#x_n_train = (x_n_train-x_n_train.min(axis = 0))/(x_n_train.max(axis = 0)-x_n_train.min(axis = 0))
y_n_label = np.array(y_label).reshape(N,1) # N,1

#

print(y_n_label)
#print(x_n_label)
#print(y_n_label)
#print(np.size(x_n_train,0))
#print(np.size(x_n_train,1))
#print(np.size(y_n_label,0))
#print(np.size(y_n_label,1))

feature_num = np.size(x_n_train,1)
print('feature_num:',feature_num)

# training


# 建立3層NN
model = Sequential()
model.add(Dense(units=10,input_shape=(feature_num,),activation='relu'))#input_dim=feature_num
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(units=256,activation='relu'))


model.add(Dense(units=1,activation='linear'))
model.summary()
# 選擇loss function和optimizing method
keras.optimizers.Adam(lr=1e-3 , beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=0.0)
model.compile(loss='mse',optimizer='Adam',metrics=['accuracy'])


#keras.initializers.Constant(value=0)
print('Training....')
#print(x_n_train[:5])
#print(y_n_label[10003:])

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 500, verbose = 1,mode = 'min')
history = model.fit(x_n_train, y_n_label, epochs=1000, batch_size=128, validation_split=0.01, shuffle=True, callbacks=[early_stopping])            
x_test = x_n_train[0].reshape(-1,feature_num)
y_test = model.predict(x_test)
x_test1 = x_n_train[1].reshape(-1,feature_num)
y_test1 = model.predict(x_test1)
x_test2 = x_n_train[2].reshape(-1,feature_num)
y_test2 = model.predict(x_test2)
print('x_test:',x_test)
print('y_test:',y_test)
print('y:',y_n_label[0])
print('x_test1:',x_test1)
print('y_test1:',y_test1)
print('y:',y_n_label[1])
print('x_test2:',x_test2)
print('y_test2:',y_test2)
print('y:',y_n_label[2])

model.save('localization_DNN256.h3')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('DNN256.png')
plt.show()
