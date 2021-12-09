import os, sys
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Add, Reshape
from keras.models import load_model,Model
import tensorflow as tf
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm as tqdm
from scipy.io import FortranFile
from scipy.interpolate import interp2d

from matplotlib import cm
#from colorspacious import cspace_converter
from collections import OrderedDict

#This part of the code creates the images of the decomposed fields
#with the already trained model

filenm='cnn2.hdf5' # File name of this model

seqs= 2
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
#var = var[:,0:ix,0:jx,:]
print(var.shape)

model = load_model(filenm)
layer_outputs = [layer.output for layer in model.layers[:]]
print(len(layer_outputs))
activation_model = Model(inputs = model.input, outputs = layer_outputs)
print(var[0:1,:,:,:].shape)
activation = activation_model.predict(var[0:1,:,:,:])
#x = model.predict(var)
mode1 = activation[45]
mode2 = activation[46]
reconstructed = model.predict(var[0:1,:,:,:])#activation[47]
print(mode1.shape)
print(mode2.shape)
f = FortranFile('geo.s8','r')
grid = f.read_reals(np.float).reshape(ix+1,jx+1,kx+1,
                                   2,order = 'F')
grid = np.squeeze(grid)

m = int(jx+1)
l2norm = np.zeros(reconstructed.shape)
l2norm = np.sqrt(np.square(var[0:1,:,:,:] - reconstructed))
# inter = interp2d(grid[:,0:m,0],grid[:,0:m,1],reconstructed[0,:,0:m,0])
# reconstructed = inter(grid[:,0:m,0],grid[:,0:m,1])
#cmaps['Diverging'] = ['seismic']

#To reverse colormap
hot1 = plt.cm.get_cmap('hot')
hot = hot1.reversed()
plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],reconstructed[0,:,0:m,0],cmap='jet',shading='gouraud')
#plt.margins(x=-.3,y=-0.35)
plt.axis('scaled')
plt.axis([-1,8,-3.5,3.5])
plt.ylabel('y',fontname = 'Times New Roman', fontsize = 20)
plt.xlabel('x',fontname = 'Times New Roman', fontsize = 20)
plt.xticks(fontname = 'Times New Roman', fontsize = 20)
plt.yticks(fontname = 'Times New Roman', fontsize = 20)
plt.tight_layout()
#plt.show()
plt.savefig('predicted_u.png')
plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],reconstructed[0,:,0:m,1],cmap='jet',shading='gouraud')
plt.savefig('predicted_v.png')

plt.ylabel('y',fontname = 'Times New Roman', fontsize = 20)
plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],var[0,:,0:m,0],cmap='jet',shading='gouraud')
plt.xlabel('x',fontname = 'Times New Roman', fontsize = 20)

#plt.imshow(var[0,:,0:m,0], interpolation = 'bilinear')
#plt.show()
plt.savefig('u.png')
plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],var[0,:,0:m,1],cmap='jet',shading='gouraud')
plt.savefig('v.png')

plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],l2norm[0,:,0:m,0],cmap=hot,shading='gouraud',vmin=0,vmax=0.1)

cb = plt.colorbar(shrink=0.75)
for l in cb.ax.yaxis.get_ticklabels():
    l.set_family("Times New Roman")
    l.set_size(14)
#cb.ticks(fontname = 'Times New Roman', fontsize = 15)
title = plt.title(r'$L_2$ norm error',fontname = 'Times New Roman', fontsize = 15)
plt.savefig('l2norm_u.png')
plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],l2norm[0,:,0:m,1],cmap=hot,shading='gouraud',vmin=0,vmax=0.1)
plt.savefig('l2norm_v.png')
plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],mode1[0,:,0:m,0],cmap='jet',shading='gouraud')
cb.remove()
title.remove()
#plt.ylabel('y',fontname = 'Times New Roman')
#plt.xlabel('x',fontname = 'Times New Roman')
#plt.show()
plt.savefig('mode1_u.png')
plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],mode2[0,:,0:m,0],cmap='jet',shading='gouraud')
#plt.ylabel('y',fontname = 'Times New Roman')
#plt.xlabel('x',fontname = 'Times New Roman')
#plt.show()
plt.savefig('mode2_u.png')
plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],mode1[0,:,0:m,1],cmap='jet',shading='gouraud')
#plt.ylabel('y',fontname = 'Times New Roman')
#plt.xlabel('x',fontname = 'Times New Roman')
#plt.show()
plt.savefig('mode1_v.png')
plt.pcolormesh(grid[:,0:m,0],grid[:,0:m,1],mode2[0,:,0:m,1],cmap='jet',shading='gouraud')
#plt.ylabel('y',fontname = 'Times New Roman')
#plt.xlabel('x',fontname = 'Times New Roman')
#plt.show()
plt.savefig('mode2_v.png')


# g = open('tec.plt')
# print('title="grid"',file=g)
# print('variables="x-grid" "y-grid" "u" "v"',file=g)
# print('zone t="grid" f=point, i=',ix+2,', j=',jx+2,file=g)
# j=0
# i=0
#   while  j<=jx+1:
#         while i<=ix+1:
#             print(geo(i,j,1),geo(i,j,2),x(0,i,j,1), var(i,j,0,2,400)
