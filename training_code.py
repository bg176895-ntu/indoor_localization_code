import numpy as np
import pandas as pd
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt
data_path='big_data_train/train_data/data/'
npy_name_list=[x for x in os.listdir(data_path)]
x_train=[]
for y in sorted(npy_name_list):
    print(y)
    load = np.load(data_path+y)
    print(np.size(load))
    x_train.append(load.reshape(-1,100))
#print(np.size(x_train[1],axis=1)) #100

label_path='big_data_train/train_data/label/'
npy_name_list=[x for x in os.listdir(label_path)]
y_label=[]
for y in sorted(npy_name_list):
    print(y)
    load = np.load(label_path+y)
    print(np.size(load))
    y_label.append(load.reshape(-1,1))
print(np.size(y_label[1],axis=1)) #1

Sum=0
N = [Sum+np.size(y_label[x],axis=0) for x in range(len(y_label))]
N =sum(N)
print('N: ',N) #2364284


feature_num = np.size(x_train[0],1)
print('feature_num:',feature_num)

x_train = np.concatenate(x_train,axis=0).reshape((N,feature_num))
y_label = np.concatenate(y_label,axis=0).reshape((N,1))


# training


# 建立6層NN
model = Sequential()
model.add(Dense(units=10,input_shape=(feature_num,),activation='relu'))#input_dim=feature_num
model.add(BatchNormalization())
#model.add(Dropout(0.3))
model.add(Dense(units=64,activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.3))
model.add(Dense(units=64,activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.3))
model.add(Dense(units=64,activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.3))
model.add(Dense(units=64,activation='relu'))


model.add(Dense(units=1,activation='linear'))
model.summary()
# 選擇loss function和optimizing method
keras.optimizers.Adam(lr=1e-3 , beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=0.0)
model.compile(loss='mse',optimizer='Adam',metrics=['accuracy'])
# checkpoint
filepath="big_data_train/DNN64_input_unit_10-{epoch:02d}-{val_loss:.2f}.h3"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#keras.initializers.Constant(value=0)
print('Training....')

#early_stopping = EarlyStopping(monitor = 'val_loss', patience = 500, verbose = 1,mode = 'min')
history = model.fit(x_train, y_label, epochs=100, batch_size=128, validation_split=0.01, shuffle=True, callbacks=callbacks_list)#, callbacks=[early_stopping])            
x_test = x_train[0].reshape(-1,feature_num)
y_test = model.predict(x_test)
x_test1 = x_train[1].reshape(-1,feature_num)
y_test1 = model.predict(x_test1)
x_test2 = x_train[2].reshape(-1,feature_num)
y_test2 = model.predict(x_test2)
print('x_test:',x_test)
print('y_test:',y_test)
print('y:',y_label[0])
print('x_test1:',x_test1)
print('y_test1:',y_test1)
print('y:',y_label[1])
print('x_test2:',x_test2)
print('y_test2:',y_test2)
print('y:',y_label[2])

model.save('big_data_train/DNN64_input_unit_10.h3')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('big_data_train/DNN64_input_unit_10.png')
#plt.show()


