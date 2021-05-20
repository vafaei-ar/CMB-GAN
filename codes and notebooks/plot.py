from __future__ import print_function, division

import matplotlib as mpl
mpl.use('agg')

import os
import pickle

import numpy as np
import pylab as plt

from utils import data_provider

amin,amax = 150,170


shfname = 'shapes/shapes{:d}_{:d}.pkl'.format(amin,amax)
if os.path.exists(shfname):
    with open(shfname, 'rb') as handle:
        shapes = pickle.load(handle)
else:
    shapes = get_shapes(mask_file='mask.npy',amin=amin,amax=amax)
    with open(shfname, 'wb') as handle:
        pickle.dump(shapes, handle, protocol=pickle.HIGHEST_PROTOCOL)

n_shapes = len(shapes)

cmbs = []
for i in range(10):
    cmbs.append(np.load('patches/'+str(i)+'.npy'))
cmbs = np.concatenate(cmbs)

imask, cmb = data_provider(batch_size = 1,
                    cmbs = cmbs,
                    shapes = shapes,
                    output_shape = (64, 64),
                    n_masks = 1)

masked_cmb = (1-imask)*cmb


fig,ax = plt.subplots(figsize=(8,8))
ax.imshow(cmb[0,:,:],cmap='jet')

plt.axis('off')

plt.subplots_adjust(0.01,0.01,0.99,0.99)
plt.savefig('cmb.jpg')
plt.close()




mm = masked_cmb[0,:,:]
mm[mm==0]=mm.max()

fig,ax = plt.subplots(figsize=(8,8))
ax.imshow(mm,cmap='jet')

plt.axis('off')
plt.subplots_adjust(0.01,0.01,0.99,0.99)
plt.savefig('masked.jpg')
plt.close()


mm = (1-imask)[0,:,:]

fig,ax = plt.subplots(figsize=(8,8))
ax.imshow(mm,cmap='jet')

plt.axis('off')
plt.subplots_adjust(0.01,0.01,0.99,0.99)
plt.savefig('mask.jpg')
plt.close()














