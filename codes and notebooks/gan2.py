from __future__ import print_function, division

import matplotlib as mpl
mpl.use('agg')

import os
import pickle

import argparse
import numpy as np
import pylab as plt

from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--lside', required=False, help='image size', type=int, default=64)
parser.add_argument('--epochs', required=False, help='epochs', type=int, default=50000)
parser.add_argument('--batch_size', required=False, help='batch size', type=int, default=32)
parser.add_argument('--nld', required=False, help='number d layers', type=int, default=0)
parser.add_argument('--nlg', required=False, help='number g layers', type=int, default=4)
parser.add_argument('--ch_scale', required=False, help='channel scale', type=float, default=1.5)
parser.add_argument('--alpha', required=False, help='alpha', type=float, default=0.01)
parser.add_argument('--kernel', required=False, help='kernel size', type=int, default=3)
parser.add_argument('--amin', required=False, help='a min', type=int, default=10)
parser.add_argument('--amax', required=False, help='a max', type=int, default=0)
parser.add_argument('--n_masks', required=False, help='n masks', type=int, default=1)
parser.add_argument('--sinterval', required=False, help='sinterval', type=int, default=200)
parser.add_argument('--analyze', action="store_true")
parser.add_argument('--restart', action="store_true")

parser.add_argument('--nfl', required=False, help='NFL', type=float, default=0)

args = parser.parse_args()
lside = args.lside
epochs = args.epochs
batch_size = args.batch_size
n_layers_dis = args.nld
n_layers_gen = args.nlg

if n_layers_dis==0:
    n_layers_dis = n_layers_gen-1

ch_scale = args.ch_scale
alpha = args.alpha
kernel_size = args.kernel
amin = args.amin
amax = args.amax

if amax==0:
    amax = amin+20

n_masks = args.n_masks
analyze = args.analyze
sample_interval = args.sinterval
try_restore = not args.restart

nfl = args.nfl

#prefix = 'models_{}_{}/l{}_g{}_d{}_ch{}_k{}'.format(amin,amax,lside,n_layers_gen,n_layers_dis,ch_scale,kernel_size)

#ch_mkdir('models_{}_{}'.format(amin,amax))

prefix = 'models_{}_{}_{:2.2f}/l{}_g{}_d{}_ch{}_k{}'.format(amin,amax,nfl,lside,n_layers_gen,n_layers_dis,ch_scale,kernel_size)

ch_mkdir('models_{}_{}_{:2.2f}'.format(amin,amax,nfl))

ch_mkdir('shapes')

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
    dd = np.load('patches/T'+str(i)+'.npy')
    if nfl!=0:
        dd = dd+nfl*dd**2
    cmbs.append(dd)
cmbs = np.concatenate(cmbs)

cmbs_test = np.load('patches/T10.npy')

print(cmbs.shape)
print(cmbs_test.shape)

prefix = str(alpha)+'/'+prefix+'/'
context_encoder = ContextEncoder(cmbs = cmbs,
                                 cmbs_test = cmbs_test,
                                 shapes = shapes,
                                 ll = lside,
                                 prefix = prefix,
                                 n_masks = n_masks,
                                 n_layers_dis = n_layers_dis,
                                 n_layers_gen = n_layers_gen,
                                 nremax = 20,
                                 ch_scale = ch_scale,
                                 kernel_size = kernel_size,
                                 learning_rate = 0.001,
                                 decay_rate = 0.9995,
                                 alpha = alpha,
                                 try_restore = try_restore)



#context_encoder.prefix = str(alpha)+'/'+prefix+'/'
#context_encoder.init_dirs()

#imask, cmb = context_encoder.data_provider_test(batch_size)
#masked_cmb = (1-imask)*cmb
#missing_parts = imask*cmb

#plt.imshow(cmb[0,:,:,0])
#plt.colorbar()
#plt.savefig('1.jpg')
#plt.close()

#plt.imshow(imask[0,:,:,0])
#plt.colorbar()
#plt.savefig('2.jpg')
#plt.close()

#plt.imshow(masked_cmb[0,:,:,0])
#plt.colorbar()
#plt.savefig('3.jpg')
#plt.close()

#plt.imshow(missing_parts[0,:,:,0])
#plt.colorbar()
#plt.savefig('4.jpg')
#plt.close()

#exit()
learning_rate = None
#learning_rate = 1e-6
context_encoder.decay_rate = 0.99985

context_encoder.train(epochs = epochs,
                      batch_size = batch_size,
                      sample_interval = sample_interval,
                      learning_rate = learning_rate)

prefix = '{}/res/mn{}_mx{}_{:2.2f}_l{}_g{}_d{}_ch{}_k{}'.format(alpha,
                                                                amin,
                                                                amax,
                                                                nfl,
                                                                lside,
                                                                n_layers_gen,
                                                                n_layers_dis,
                                                                ch_scale,
                                                                kernel_size)
ch_mkdir('{}/res'.format(alpha))

if analyze:
    for i in range(10,15):
        print('TEST '+str(i))
        cl_path = 'cls/cl_{}.npy'.format(i)
        clw_path = 'cls/clwh_{}_{}_{:2.2f}_{}.npy'.format(amin,amax,nfl,i)
        file_path = 'patches/T{}.npy'.format(i)
        postfix = '_{}'.format(i)
        
#        if os.path.exists(prefix+'power'+postfix+'.jpg'):
#            continue

        context_encoder.gan_analyze(cl_path = cl_path,
                                    clw_path = clw_path,
                                    file_path = file_path,
                                    prefix = prefix,
                                    postfix = postfix)














