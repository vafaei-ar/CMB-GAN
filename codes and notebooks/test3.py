from __future__ import print_function, division

import matplotlib as mpl
mpl.use('agg')

import os
import pickle

import argparse
import numpy as np
import pylab as plt

from utils import *



from scipy.ndimage.filters import gaussian_filter
from ccgpack import patch2sky, sky2patch
import healpy as hp



parser = argparse.ArgumentParser()
#parser.add_argument('--num', required=False, help='test number', type=int, default=0)
parser.add_argument('--lside', required=False, help='image size', type=int, default=256)
parser.add_argument('--nld', required=False, help='number d layers', type=int, default=0)
parser.add_argument('--nlg', required=False, help='number g layers', type=int, default=3)
parser.add_argument('--ch_scale', required=False, help='channel scale', type=float, default=1.5)
parser.add_argument('--alpha', required=False, help='alpha', type=float, default=0.01)
parser.add_argument('--kernel', required=False, help='kernel size', type=int, default=3)

parser.add_argument('--mamin', required=False, help='a min', type=int, default=10)
parser.add_argument('--mamax', required=False, help='a max', type=int, default=0)

parser.add_argument('--tamin', required=False, help='a min', type=int, default=0)
parser.add_argument('--tamax', required=False, help='a max', type=int, default=20)

parser.add_argument('--nfl', required=False, help='NFL', type=float, default=0)

args = parser.parse_args()
#num = args.num
lside = args.lside
n_layers_dis = args.nld
n_layers_gen = args.nlg

if n_layers_dis==0:
    n_layers_dis = n_layers_gen-1

ch_scale = args.ch_scale
alpha = args.alpha
kernel_size = args.kernel
mamin = args.mamin
mamax = args.mamax
tamin = args.tamin
tamax = args.tamax

nfl = args.nfl

if mamax==0:
    mamax = mamin+20

prefix = 'models_{}_{}_{:2.2f}/l{}_g{}_d{}_ch{}_k{}'.format(mamin,mamax,nfl,64,n_layers_gen,n_layers_dis,ch_scale,kernel_size)

prefix = str(alpha)+'/'+prefix+'/'

#shfname = 'shapes/shapes{:d}_{:d}.pkl'.format(mamin,mamax)
#with open(shfname, 'rb') as handle:
#    shapes = pickle.load(handle)
patches = np.load('mask.npy')
shapes = []

#python test3.py --alpha 0.01 --nfl 1000 --mamin 290 --tamin 500 --tamax 1000

##### ORIGINAL MASK
#new_mask = []
#for i in range(patches.shape[0]):
#    blobs = measure.label(patches[i]==0)
#    bmax = blobs.max()
#    for j in range(1,bmax+1):
#        if np.sum(blobs==j)>tamax or np.sum(blobs==j)<tamin :
#            shapes.append(trim_zeors(blobs,j))
#            blobs[blobs==j] = 0
#    new_mask.append(blobs==0)
#masks = np.array(new_mask)


##### Produced MASK
shfname = 'shapes/shapes{:d}_{:d}.pkl'.format(tamin,tamax)
if os.path.exists(shfname):
    with open(shfname, 'rb') as handle:
        shapes = pickle.load(handle)
else:
    shapes = get_shapes(mask_file='mask.npy',amin=tamin,amax=tamax)
    with open(shfname, 'wb') as handle:
        pickle.dump(shapes, handle, protocol=pickle.HIGHEST_PROTOCOL)

ll = lside
l = 2048//ll


for num in range(10,15):
    prefix2 = '{}/res/n{}_tn{}_tx{}_mn{}_mx{}_{:2.2f}_l{}_g{}_d{}_ch{}_k{}_'.format(alpha,
                                                                                    num,
                                                                                    tamin,
                                                                                    tamax,
                                                                                    mamin,
                                                                                    mamax,
                                                                                    nfl,
                                                                                    lside,
                                                                                    n_layers_gen,
                                                                                    n_layers_dis,
                                                                                    ch_scale,
                                                                                    kernel_size)

    cmbs_test = np.load('patches/T'+str(num)+'.npy')
    
    if nfl!=0:
        cmbs_test = cmbs_test+nfl*cmbs_test**2

    context_encoder = ContextEncoder(cmbs = cmbs_test,
                                     cmbs_test = cmbs_test,
                                     shapes = shapes,
                                     ll = lside,
                                     prefix = prefix,
                                     n_masks = 1,
                                     n_layers_dis = n_layers_dis,
                                     n_layers_gen = n_layers_gen,
                                     nremax = 20,
                                     ch_scale = ch_scale,
                                     kernel_size = kernel_size,
                                     learning_rate = 0.001,
                                     decay_rate = 0.9995,
                                     try_restore = 0)

    context_encoder.load_generator()

    cmbs = cmbs_test
    cmbs = cmbs-cmbs.min()
    cmbs = cmbs/cmbs.max()

    plt.imshow(cmbs[0],cmap='jet')
    plt.savefig(prefix2+'cmb.jpg',dpi=150)
    plt.close()

    lp = 2048//64
    mask_pp = np.ones((12*lp*lp, 64, 64))
    for i in range(0,12*lp*lp):
        imask = mask_maker(shapes = shapes,
                           output_shape = (64,64),
                           n_masks = 1)
        mask_pp[i] = (1-imask)
        
    masks = np.zeros((12, 2048, 2048))
    for k in range(12):
        masks[k] = deblock(mask_pp[k*lp*lp:(k+1)*lp*lp,:,:])

    mask_pp = []
    for i in range(12):
         mask_pp.append(blocker(masks[i], 256, 256))

    mask_pp = np.concatenate(mask_pp)

    cl_path = 'cls/cl_{}.npy'.format(i)
    if not os.path.exists(cl_path):
        fullsky = patch2sky(cmbs)
        fullsky = hp.pixelfunc.reorder(fullsky, inp=None, out=None, r2n=None, n2r=1)
        cl = hp.anafast(fullsky)
        np.save(cl_path[:-4],cl)
    else:
        cl = np.load(cl_path)

    clw_path = 'cls/clwh_{}_{}_{:2.2f}_{}.npy'.format(mamin,mamax,nfl,i)
    if not os.path.exists(clw_path):
        fullsky = cmbs*masks
        fullsky = patch2sky(fullsky)
        fullsky[fullsky==0] = cmbs.mean()
        fullsky = hp.pixelfunc.reorder(fullsky, inp=None, out=None, r2n=None, n2r=1)
        clw = hp.anafast(fullsky)
        np.save(clw_path[:-4],clw)
    else:
        clw = np.load(clw_path)

    fsg = hp.sphtfunc.synfast(cls=cl, nside=2048)
    fsg = hp.pixelfunc.reorder(fsg, inp=None, out=None, r2n=1, n2r=None)
    
    fsg = sky2patch(fsg)
    fsg = fsg-fsg.min()
    fsg = fsg/fsg.max()
    
    fsg = cmbs*masks+fsg*(1-masks)
    
    plt.imshow(fsg[0],cmap='jet')
    plt.savefig(prefix2+'fsg.jpg',dpi=150)
    plt.close()
#    fsg = patch2sky(fsg)
#    hp.mollview(fsg,cmap='jet',title='Inpainted',nest=1)
#    plt.subplots_adjust(left=0.01,bottom=0.05,right=0.99,top=0.99,wspace=0.01,hspace=0.01)
#    plt.savefig('maps.jpg',dpi=150)
#    plt.close()
#    exit()

    ell = np.arange(len(cl))
    dl = ell * (ell + 1) * cl
    dlw = ell * (ell + 1) * clw

    cmbs_pp = []
    for i in range(12):
         cmbs_pp.append(blocker(cmbs[i], ll, ll))

    cmbs_pp = np.concatenate(cmbs_pp)

    mask_pp = []
    for i in range(12):
         mask_pp.append(blocker(masks[i], ll, ll))

    mask_pp = np.concatenate(mask_pp)

    imasks = 1-mask_pp
    masked_cmb = cmbs_pp*mask_pp

    imasks = np.array(imasks)[:,:,:,None]
    masked_cmb = np.array(masked_cmb)[:,:,:,None]

    gen_whole = context_encoder.generator.predict([masked_cmb,imasks])
    
    cmbs3 = np.zeros((12, 2048, 2048))

    res1 = []
    res2 = []
    res3 = []

    thresholds = [0.25,0.5,0.75]
    for k in range(12):
        cmbs3[k] = deblock(gen_whole[k*l*l:(k+1)*l*l,:,:,0])

        bins,hist,nu,n1,ksis = eval_ksi(cmbs[k],
                                        thresholds = thresholds,
                                        dmin=0,
                                        dmax=1,
                                        nu_num=100,
                                        peak=True,
                                        rmax=300,
                                        crand=5)
        res1.append([bins,hist,nu,n1,ksis])

        bins,hist,nu,n1,ksis = eval_ksi(fsg[k],
                                        thresholds = thresholds,
                                        dmin=0,
                                        dmax=1,
                                        nu_num=100,
                                        peak=True,
                                        rmax=300,
                                        crand=5)
        res2.append([bins,hist,nu,n1,ksis])
        
        bins,hist,nu,n1,ksis = eval_ksi(cmbs3[k],
                                        thresholds = thresholds,
                                        dmin=0,
                                        dmax=1,
                                        nu_num=100,
                                        peak=True,
                                        rmax=300,
                                        crand=5)
        res3.append([bins,hist,nu,n1,ksis])

    plt.imshow(cmbs3[0],cmap='jet')
    plt.savefig(prefix2+'gen.jpg',dpi=150)
    plt.close()
        
    np.save(prefix2+'res1',res1)
    np.save(prefix2+'res2',res2)
    np.save(prefix2+'res3',res3)

    fullsky3 = patch2sky(cmbs3)
    fullsky3 = hp.pixelfunc.reorder(fullsky3, inp=None, out=None, r2n=None, n2r=1)

#    np.save('cmb_masked_'+str(num),cmbs*masks)
#    np.save('cmb_gen_'+str(num),cmbs3)
#    continue

    cl3 = hp.anafast(fullsky3)
    dl3 = ell * (ell + 1) * cl3


    np.save(prefix2+'clpred',cl3)

    plt.figure(figsize=(8, 5))
    plt.plot(ell, dl,color='b',ls='none',marker='.',alpha=0.3)
    plt.plot(ell, dl3,color='r',ls='none',marker='.',alpha=0.3)
    plt.plot(ell, gaussian_filter(dl,10),color='b')
    plt.plot(ell, gaussian_filter(dl3,10),color='r')

    plt.xlabel("$\ell$",fontsize=20)
    plt.ylabel("$\ell(\ell+1)C_{\ell} [K]$",fontsize=15)
    plt.xscale('log')
    plt.xlim(1,2500)
    plt.grid()
    plt.subplots_adjust(left=0.12,bottom=0.13,right=0.99,top=0.99,wspace=0.01,hspace=0.01)
    plt.savefig(prefix2+'power.jpg',dpi=150)
    plt.close()

    #fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

    #plt.sca(ax1)
    #hp.mollview(fullsky,cmap='jet',title='Truth',hold=1)
    #plt.sca(ax2)
    #hp.mollview(fullsky3,cmap='jet',title='Inpainted',hold=1)
    #plt.subplots_adjust(left=0.01,bottom=0.05,right=0.99,top=0.99,wspace=0.01,hspace=0.01)
    #plt.savefig(prefix2+'maps.jpg',dpi=150)
    #plt.close()

    xi2pred = np.sum((dl-dl3)**2)
    xi2w = np.sum((dl-dlw)**2)
    print(xi2pred/xi2w)




