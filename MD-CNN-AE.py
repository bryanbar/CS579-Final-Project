import os, sys
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Add, Reshape,ZeroPadding2D
from keras.models import Model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.io import FortranFile


import urllib.request

#########################################
# 1. Set parameter
#########################################
n_epoch=2000#2000 # Number of epoch
pat=30#50 # Patience
filenm='cnn2' # File name of this model

#########################################
# 2. Read data from fortran .s8 files (binary files)
#########################################
#number of archives
seqs= 11
#size of computational domain
ix = 193
jx = 129
kx = 0
ivar = 2
ntime = 512
ntime1= 512
ntime2 = ntime1 + ntime*(seqs-2)
shape = (ix+1,jx+1,1,ivar,ntime2)

print('Number of snapshots =',ntime2)

var = np.zeros(shape)

print('Reading from prim files')
n = 2

while (n<=seqs):
    fname = 'prim_'
    ext = '.s8'
    num = str(n)
    if len(num)==1:
        fname = fname + '0'+ num + ext
    else:
        fname = fname + num + ext
    print('Opening ',fname)
    f = FortranFile(fname,'r')
    if n==2:
        b = f.read_reals(np.float).reshape(ix+1,jx+1,kx+1,
                                           ivar,ntime1,order = 'F')
        var[:,:,:,:,0:ntime1] = b
        last = ntime1
    else:
        b = f.read_reals(np.float).reshape(ix+1,jx+1,kx+1,
                                           ivar,ntime,order = 'F')
        var[:,:,:,:,last:ntime+last] = b
        last = ntime + last
        print(last)

    n = n+1
var = np.squeeze(var)
var = np.moveaxis(var,3,0)

#If needed, decomposed field can be vizualized with less snapshots by
#removing the mean-flow


##compute average mean-flow
# shapemf = (ix+1,jx+1,2)
# meanf = np.zeros(shapemf)
#
#
# i = 0
# while (i <= ix):
#     j = 0
#     while (j <= jx):
#         nt = 0
#         meanu = np.sum(var[:,i,j,0])/ntime2
#         meanv = np.sum(var[:,i,j,1])/ntime2
#         var[:,i,j,0] = var[:,i,j,0] - meanu
#         var[:,i,j,1] = var[:,i,j,1] - meanv
#         meanf[i,j,0] = meanu/ntime2
#         meanf[i,j,1] = meanv/ntime2
#         # while (nt<ntime2):
#         #     var[nt,i,j,0] = var[nt,i,j,0] - meanu
#         #     var[nt,i,j,1] = var[nt,i,j,1] - meanv
#         #     nt = nt + 1
#         j = j + 1
#     i = i + 1
#
#
# f = FortranFile('geo.s8','r')
# grid = f.read_reals(np.float).reshape(ix+1,jx+1,kx+1,
#                                    2,order = 'F')
# grid = np.squeeze(grid)
# m = int(jx+1)
# plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],meanf[:,0:m,0],cmap='jet',shading='gouraud')
# plt.margins(x=-.3,y=-0.35)
# plt.ylabel('y',fontname = 'Times New Roman', fontsize = 20)
# plt.xlabel('x',fontname = 'Times New Roman', fontsize = 20)
# plt.xticks(fontname = 'Times New Roman', fontsize = 20)
# plt.yticks(fontname = 'Times New Roman', fontsize = 20)
# plt.tight_layout()
# plt.savefig('u_mean.png')
# plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],meanf[:,0:m,1],cmap='jet',shading='gouraud')
# plt.savefig('v_mean.png')
# plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],var[100,:,0:m,0],cmap='jet',shading='gouraud')
# plt.savefig('u_fluctuations.png')
# plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],var[100,:,0:m,1],cmap='jet',shading='gouraud')
# plt.savefig('v_fluctuations.png')

#########################################
# 4. Autoencoder
#########################################
input_img = Input(shape=(ix+1, jx+1, 2))
## Encoder
x = Conv2D(16, (3, 3), activation='tanh', padding='valid')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Reshape([3*2*4])(x)
encoded = Dense(2,activation='tanh')(x)
## Two variables
val1= Lambda(lambda x: x[:,0:1])(encoded)
val2= Lambda(lambda x: x[:,1:2])(encoded)
## Decoder 1
x1 = Dense(3*2*4,activation='tanh')(val1)
x1 = Reshape([3,2,4])(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(4,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(16,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)

x1 = ZeroPadding2D(1)(x1)
x1d = Conv2D(2,(3,3),activation='linear',padding='same')(x1)
## Decoder 2
x2 = Dense(3*2*4,activation='tanh')(val2)
x2 = Reshape([3,2,4])(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(4,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(16,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = ZeroPadding2D(1)(x2)
x2d = Conv2D(2,(3,3),activation='linear',padding='same')(x2)

decoded = Add()([x1d,x2d])

autoencoder = Model(input_img, decoded)
opt = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
autoencoder.compile(optimizer='adam', loss='mse')

# Check the network structure
autoencoder.summary()

#########################################
# 5. Model training
#########################################
tempfn='./'+filenm+'.hdf5'
model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=pat,verbose=1)
cb = [model_cb, early_cb]

X_train,X_test,y_train,y_test=train_test_split(var,var,test_size=0.3,random_state=1)

# Run inference on CPU
#with tf.device('/cpu:0'):
history=autoencoder.fit(X_train, y_train,
                epochs=n_epoch,
                batch_size=110,
                shuffle=True,
                validation_data=(X_test, y_test),
                callbacks=cb )

df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
tempfn='./'+filenm+'.csv'
df_results.to_csv(path_or_buf=tempfn,index=False)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
