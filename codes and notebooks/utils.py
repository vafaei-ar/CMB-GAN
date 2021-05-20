import matplotlib as mpl
# mpl.use('agg')

import numpy as np
from skimage import measure
from scipy.ndimage import rotate
from ccgpack import ch_mkdir

import os
import pickle
import pylab as plt
from itertools import product

from scipy.ndimage.filters import gaussian_filter
from ccgpack import patch2sky, sky2patch, ffcf_no_random#,pdf
import healpy as hp

# from keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Lambda
from tensorflow.keras.layers import multiply, add, Multiply, Add
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

import os
import numpy as np
from tqdm import tqdm,tqdm_notebook,trange

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    return True

def pdf(m,bins=20,normed=None):
    m = np.array(m)
    hist, bin_edges = np.histogram(m, bins, density=normed)
    bins = 0.5*(bin_edges[1:]+bin_edges[:-1])
    return bins,hist

def blocker(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
def deblock(pp):
    l = int(np.sqrt(pp.shape[0]))
    ll = pp.shape[1]
    p = np.zeros((l*ll,l*ll))
    for i in range(l):
        for j in range(l):
            p[ll*j:ll*(j+1),ll*i:ll*(i+1)] = pp[i+l*j,:,:]
    return p

def trim_zeors(blobs,ind):
    x = blobs==ind
    filty = np.sum(x,axis=0)!=0
    x = x[:,filty]
    filtx = np.sum(x,axis=1)!=0
    x = x[filtx,:]
    return x

def get_shapes(mask_file,amin=0,amax=40):
    patches = np.load(mask_file)
    shapes = []
    for i in range(patches.shape[0]):
        blobs = measure.label(patches[i]==0)
        bmax = blobs.max()
        for j in range(1,bmax+1):
            if np.sum(blobs==j)<=amax and np.sum(blobs==j)>=amin:
                shapes.append(trim_zeors(blobs,j))
    return shapes

def N1(d,dmin=None,dmax=None,num=100,gt=True):

    if dmin is None:
        dmin = d.min()
    if dmax is None:
        dmax = d.max()

    nu = np.linspace(dmin,dmax,num)
    n1 = []
    for i in nu:
        if gt:
            n1.append(np.mean(d>i))
        else:
            n1.append(np.mean(d<i))
    n1 = np.array(n1)
    return nu,n1


def exterma(arr,peak=True):
    
    dim = len(arr.shape)       # number of dimensions
    offsets = [0, -1, 1]     # offsets, 0 first so the original entry is first 
    filt = np.ones(arr.shape,dtype=np.int8)
    for shift in product(offsets, repeat=dim):
        if np.all(np.array(shift)==0):
            continue
    #    print(shift)
    #    print(np.roll(b, shift, np.arange(dim)))
        rolled = np.roll(arr, shift, np.arange(dim))
        
        if peak:
            filt = filt*(arr>rolled)
        else:
            filt = filt*(arr<rolled)
            
    return filt     

class PPTT:
    
    def __init__(self,mm,peak,mask=None):
    
#        mm = mm[:512,:512]
#        mask = mask[:512,:512]
    
        self.nside = mm.shape[0]
        peaks = exterma(mm ,peak=peak)
        self.peak_img = mm+0
        self.peak_img[np.logical_not(peaks.astype(bool))] = 0

        plt.imshow(mm)
        plt.savefig('map.jpg')
        
        plt.imshow(self.peak_img)
        plt.savefig('peaks.jpg')
        plt.close()
        if mask is None:
            self.mask = np.ones(mm.shape)
        else:
            self.mask = mask

    def pptt(self,th,rmax=300,crand=5):
        self.dim = rmax
        mc2 = self.peak_img+0
        mc2[mc2<th] = 0

        nf1 = np.argwhere(mc2).T
        n_peaks = nf1.shape[1]
        
        if n_peaks!=0:
            nnn = crand*n_peaks
            rlist = np.random.randint(0,self.nside,(nnn,2))
            rimg = np.zeros(mc2.shape)
            rows, cols = zip(*rlist)
            rimg[rows, cols] = 1
            rimg = rimg*self.mask
            rlist = np.argwhere(rimg).T
            ksi = ffcf_no_random(fl1=nf1, fl2=nf1, rlist=rlist, rmax=rmax)
            return ksi
        else:
            return np.zeros(self.dim)

def eval_ksi(m,thresholds,dmin=None,dmax=None,nu_num=100,peak=True,rmax=300,crand=5):
    bins,hist = pdf(m[np.isfinite(m)].reshape(-1),bins=200,normed=1)    
    pptt = PPTT(m,peak=peak)
    peaks = pptt.peak_img[pptt.peak_img!=0]
    nu,n1 = N1(peaks,dmin=dmin,dmax=dmax,num=nu_num,gt=peak)
    
    print(np.sum(pptt.peak_img!=0),np.sum(pptt.peak_img==0))

    ksis = []
    for th in thresholds:
        ksi = pptt.pptt(th,rmax=rmax,crand=crand)
        ksis.append(ksi)
    return bins,hist,nu,n1,ksis


def mask_maker(shapes,output_shape,n_masks):
    n_shapes = len(shapes)
    mask = np.zeros(output_shape)

    for _ in range(n_masks):
        patch = shapes[np.random.randint(n_shapes)]
        
        cond = max(patch.shape)>max(output_shape)
        while cond:
            patch = shapes[np.random.randint(n_shapes)]
            cond = max(patch.shape)>max(output_shape)
            
    #     patch = rotate(patch,
    #                    angle=np.random.uniform(0,360),
    #                    axes=(1, 0),
    #                    reshape=True,
    #                    order=3)
        patch = np.rot90(patch,
                         k = np.random.randint(4))

        pshape = patch.shape
        patch = patch>0

        i0 = np.random.randint(output_shape[0]-pshape[0])
        j0 = np.random.randint(output_shape[1]-pshape[1])

        mask[i0:i0+pshape[0],j0:j0+pshape[1]] = patch
    return mask

def random_cut(pcmb,output_shape):
    pshape = pcmb.shape
    
    i0 = np.random.randint(pshape[0]-output_shape[0])
    j0 = np.random.randint(pshape[1]-output_shape[1])

    return pcmb[i0:i0+output_shape[0],j0:j0+output_shape[1]]

def data_provider(batch_size,
                  cmbs,
                  shapes,    
                  output_shape = (32,32),
                  n_masks = 2):
    x = []
    y = []
    
    for _ in range(batch_size):
        i_rand = np.random.randint(cmbs.shape[0])
        pcmb = cmbs[i_rand]

        pcmb_cut = random_cut(pcmb,output_shape)

        imask = mask_maker(shapes = shapes,
                          output_shape = output_shape,
                          n_masks = n_masks)

        # x.append((1-mask)*pcmb_cut)
        x.append(imask)
        y.append(pcmb_cut)
        
    return np.array(x),np.array(y)

def power_spectrum(m,size):
    nside = m.shape[0]
    
    mk = np.fft.fft2(m)
    kmax = int(1.5*nside)
    power = np.zeros(kmax)
    nn = np.zeros(kmax)
    for i in range(nside):
        for j in range(nside):
            k = int(np.sqrt(i**2+j**2))
            power[k] += np.absolute(mk[i,j])**2
            nn[k] += 1
            
    filt = nn!=0
    power[filt] = power[filt]/nn[filt]
    ls = (np.arange(1,kmax)+1)*360./size
    return ls,power[1:]*(ls*(ls+1))/(2*np.pi)/((nside*np.pi)**2)



def blocker(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
def deblock(pp):
    l = int(np.sqrt(pp.shape[0]))
    ll = pp.shape[1]
    p = np.zeros((l*ll,l*ll))
    for i in range(l):
        for j in range(l):
            p[ll*j:ll*(j+1),ll*i:ll*(i+1)] = pp[i+l*j,:,:]
    return p


class ContextEncoder():
    def __init__(self,
                 cmbs,
                 cmbs_test,
                 shapes,
                 ll,
                 prefix,
                 n_masks=2,
                 n_layers_dis=2,
                 n_layers_gen=2,
                 nremax = 20,
                 ch_scale=2,
                 kernel_size=3,
                 learning_rate = 0.0001,
                 decay_rate = 0.9992,
                 alpha = 0.01,  #GAN importance
                 try_restore=False):
                 
        self.img_rows = ll
        self.img_cols = ll
        self.n_masks = n_masks
        self.n_layers_dis = n_layers_dis
        self.n_layers_gen = n_layers_gen
        self.nremax = nremax
        self.ch_scale = ch_scale
        self.kernel_size = kernel_size
        
        # self.mask_height = 8
        # self.mask_width = 8
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.missing_shape = (self.img_rows, self.img_cols, self.channels)#(self.mask_height, self.mask_width, self.channels)
        self.cmbs = cmbs
        self.cmbs = self.cmbs-self.cmbs.min()
        self.cmbs = self.cmbs/self.cmbs.max()
        self.cmbs_test = cmbs_test
        self.cmbs_test = self.cmbs_test-self.cmbs_test.min()
        self.cmbs_test = self.cmbs_test/self.cmbs_test.max()
        self.shapes = shapes
        self.prefix = prefix+'/'
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

        self.init_dirs()

        optimizer = Adam(self.learning_rate, 0.5)

        # Building models
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        # compiling models
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # The generator takes noise as input and generates the missing
        # part of the image
        masked_img = Input(shape=self.img_shape)
        imask_its = Input(shape=self.img_shape)
        gen_whole = self.generator([masked_img,imask_its])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines
        # if it is generated or if it is a real image
        valid = self.discriminator(gen_whole)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(inputs=[masked_img,imask_its] , outputs=[gen_whole, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[1-alpha,alpha],
            optimizer=optimizer)

        # self.shapes = get_shapes(mask_file='../dataset/mask.npy',amax=40)
        # n_shapes = len(shapes)

        # cmbs = []
        # for i in range(10):
        #     cmbs.append(np.load('../dataset/patches/'+str(i)+'.npy'))
        # self.cmbs = np.concatenate(cmbs)

        self.epoch0 = 0
        self.nres = np.zeros((0,2))
        if try_restore:
            try:
                self.load_model()
                self.epoch0,self.learning_rate = np.load(self.prefix+'model/props.npy')
                self.nres =  np.load(self.prefix+'model/nres.npy')
                print('Model is restored from epoch {} successfully!'.format(self.epoch0))
                print('The learning rate is {}.'.format(self.learning_rate))
            except:
                print('Model is not restored, call 911!')
                pass

    def init_dirs(self):
        ch_mkdir(self.prefix)
        ch_mkdir(self.prefix+'model')
        ch_mkdir(self.prefix+'images')

    def data_provider(self,n):
        x,y = data_provider(batch_size = n,
                            cmbs = self.cmbs,
                            shapes = self.shapes,
                            output_shape = (self.img_rows, self.img_cols),
                            n_masks = self.n_masks)
        return x[:,:,:,None],y[:,:,:,None]

    def data_provider_test(self,n):
        x,y = data_provider(batch_size = n,
                            cmbs = self.cmbs_test,
                            shapes = self.shapes,
                            output_shape = (self.img_rows, self.img_cols),
                            n_masks = self.n_masks)
        return x[:,:,:,None],y[:,:,:,None]

    def build_generator(self):


        # model = Sequential(name='generator')

        # # Encoder
        # model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))

        # model.add(Conv2D(512, kernel_size=1, strides=2, padding="same"))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.5))

        # # Decoder
        # model.add(UpSampling2D())
        # model.add(Conv2D(128, kernel_size=3, padding="same"))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(UpSampling2D())
        # model.add(Conv2D(64, kernel_size=3, padding="same"))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization(momentum=0.8))

        # model.add(UpSampling2D())
        # model.add(Conv2D(64, kernel_size=3, padding="same"))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization(momentum=0.8))

        # model.add(UpSampling2D())
        # model.add(Conv2D(64, kernel_size=3, padding="same"))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization(momentum=0.8))

        # model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        # model.add(Activation('tanh'))

        # model.summary()

        # masked_img = Input(shape=self.img_shape)
        # gen_missing = model(masked_img)

        masked_img = Input(shape=self.img_shape)

        x = Conv2D(16, kernel_size=self.kernel_size, strides=2, input_shape=self.img_shape, padding="same")(masked_img)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        strides = [2,1,1,1,1,1,1]

        nch = 32
        for i in range(self.n_layers_gen):
            x = Conv2D(nch, kernel_size=self.kernel_size, strides=strides[i], padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(momentum=0.8)(x)
            nch = int(self.ch_scale*nch)

        # x = Conv2D(512, kernel_size=1, strides=2, padding="same")(x)
        # x = LeakyReLU(alpha=0.2)(x)
        # x = Dropout(0.5)(x)

        # Decoder
        x = UpSampling2D()(x)
        x = Conv2D(nch, kernel_size=self.kernel_size, padding="same")(x)
        x = Activation('relu')(x)
        x = BatchNormalization(momentum=0.8)(x)

        for i in range(self.n_layers_gen):
            if strides[i]==2:
                x = UpSampling2D()(x)
            x = Conv2D(nch, kernel_size=self.kernel_size, padding="same")(x)
            x = Activation('relu')(x)
            x = BatchNormalization(momentum=0.8)(x)
            nch = int(nch//self.ch_scale)

        x = Conv2D(self.channels, kernel_size=self.kernel_size, padding="same")(x)
        gen_whole = Activation('tanh')(x)
        
        mask_its = Input(shape=self.img_shape)

#        gen_missing = multiply([gen_whole,mask_its])
#        gen_whole = add([masked_img,gen_missing])

        gen_missing = Multiply()([gen_whole,mask_its])
        gen_whole2 = Add()([masked_img,gen_missing])
        
#        gen_whole = masked_img+(gen_whole*mask_its)

        model = Model(inputs=[masked_img,mask_its], outputs=[gen_whole2], name='generator')
        model.summary()

        return model

    def build_discriminator(self):

        img = Input(shape=self.missing_shape)
        x = Conv2D(64, kernel_size=self.kernel_size, strides=2, input_shape=self.missing_shape, padding="same")(img)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        nch = 64
        for _ in range(self.n_layers_dis):
            x = Conv2D(nch, kernel_size=self.kernel_size, strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(momentum=0.8)(x)
            nch = int(self.ch_scale*nch)

        x = Flatten()(x)
        pred = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=img, outputs=pred, name='discriminator')
        # model = Sequential(name='discriminator')
        model.summary()
        validity = model(img)

        return Model(img, validity)

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        masked_imgs = np.empty_like(imgs)
        missing_parts = np.empty_like(imgs) #np.empty((imgs.shape[0], self.mask_height, self.mask_width, self.channels))
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            missing_parts[i] = img.copy() #masked_img[_y1:_y2, _x1:_x2, :].copy()
            masked_img[_y1:_y2, _x1:_x2, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs, missing_parts, (y1, y2, x1, x2)



    def train(self, epochs, batch_size=128, sample_interval=50, learning_rate=None):

        if learning_rate is None:
            pass
        else:
            self.learning_rate = learning_rate
            print('Learning rate is renewed!')
            
        K.set_value(self.combined.optimizer.learning_rate, self.learning_rate) 
        K.set_value(self.discriminator.optimizer.learning_rate, self.learning_rate)
        # # Load the dataset
        # (X_train, y_train), (_, _) = cifar10.load_data()

        # # Extract dogs and cats
        # X_cats = X_train[(y_train == 3).flatten()]
        # X_dogs = X_train[(y_train == 5).flatten()]
        # X_train = np.vstack((X_cats, X_dogs))

        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        n_dis = 1
        n_gen = 1

        # for epoch in tqdm_notebook(range(epochs)):
        for epoch in range(int(self.epoch0),epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]

            # masked_imgs, missing_parts, _ = self.mask_randomly(imgs)

            imask, cmb = self.data_provider(batch_size)
            masked_cmb = (1-imask)*cmb
            missing_parts = imask*cmb


            for _ in range(n_dis):
                # Generate a batch of new images
                gen_whole = self.generator.predict([masked_cmb,imask])

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(cmb, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_whole, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            for _ in range(n_gen):
                g_loss = self.combined.train_on_batch([masked_cmb,imask], [missing_parts, valid])

            n_re = min(int(d_loss[0]/g_loss[1]),self.nremax)
            if n_re>=1:
                n_dis = n_re
                n_gen = 1

            n_re = min(int(g_loss[1]/d_loss[0]),self.nremax)
            if n_re>1:
                n_dis = 1
                n_gen = n_re

            self.nres = np.concatenate([self.nres,  np.array([[g_loss[1],d_loss[0]]])  ])

            # Plot the progress
            if epoch % sample_interval == 0:
                print ("%d [D loss: %f, acc: %.2f%%] [mse: %f, G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
                print('Number of generator/discriminator train {}/{}'.format(n_gen,n_dis))
                

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # idx = np.random.randint(0, X_train.shape[0], 6)
                # imgs = X_train[idx]
                self.sample_images(epoch)
                self.save_model()

                np.save(self.prefix+'model/props',np.array([epoch,self.learning_rate]))
                np.save(self.prefix+'model/nres',self.nres)            
            
            self.learning_rate = self.learning_rate*self.decay_rate
            K.set_value(self.combined.optimizer.learning_rate, self.learning_rate) 
            K.set_value(self.discriminator.optimizer.learning_rate, self.learning_rate)


        self.sample_images(epoch)
        self.save_model()

    def sample_images(self, epoch):
        r, c = 3, 6

        # masked_imgs, missing_parts, (y1, y2, x1, x2) = self.mask_randomly(imgs)
        # masked_imgs, missing_parts = self.data_provider(6)

        imask, cmb = self.data_provider_test(6)
        masked_cmb = (1-imask)*cmb
        missing_parts = imask*cmb

        # gen_missing = self.generator.predict(masked_imgs)
        gen_whole = self.generator.predict([masked_cmb,imask])

        # missing_parts = 0.5 * missing_parts + 0.5
        # masked_imgs = 0.5 * masked_imgs + 0.5
        # gen_missing = 0.5 * gen_missing + 0.5

        fig, axs = plt.subplots(r, c, figsize=(14,6))
        for i in range(c):
            axs[0,i].imshow(cmb[i,:,:,0],cmap='jet')
            axs[0,i].axis('off')
            axs[0,i].set_title('truth')

            # filled_in = (1-imask[i])*cmb[i]+imask[i]*gen_whole[i]
            # axs[1,i].imshow(filled_in[:,:,0])
            # axs[1,i].axis('off')
            # axs[1,i].set_title('inpainted-missing')

            axs[1,i].imshow(gen_whole[i,:,:,0],cmap='jet')
            axs[1,i].axis('off')
            axs[1,i].set_title('inpainted')

            axs[2,i].imshow(masked_cmb[i,:,:,0],cmap='jet')
            axs[2,i].axis('off')
            axs[2,i].set_title('masked')
            
        fig.savefig(self.prefix+"images/%d.png" % epoch)
        plt.close()

    def save_model(self):
        self.generator.save(self.prefix+'model/generator.h5')
        self.discriminator.save(self.prefix+'model/discriminator.h5')
        
        self.generator.save_weights(self.prefix+'model/w_generator.h5')
        self.discriminator.save_weights(self.prefix+'model/w_discriminator.h5')

    def load_model(self):
        try:
            self.generator.load_weights(self.prefix+'model/w_generator.h5')
            self.discriminator.load_weights(self.prefix+'model/w_discriminator.h5')
            print('Weights are recovered!')
        except:
            self.generator = load_model(self.prefix+'model/generator.h5')
            self.discriminator = load_model(self.prefix+'model/discriminator.h5')
            print('Models are recovered!')

    def load_generator(self, path=None):
        if path is None:
            path = self.prefix+'model/w_generator.h5'
        self.generator.load_weights(self.prefix+'model/w_generator.h5')
        print('Generator weights are recovered!')



    def gan_analyze(self, cl_path, 
                          clw_path,
                          file_path = None, 
                          prefix = None, 
                          postfix='',
                          nfl = 0):

        from scipy.ndimage.filters import gaussian_filter
        from ccgpack import patch2sky, sky2patch
        import healpy as hp

        ll = self.img_rows
        l = 2048//ll
        if prefix is None:
            prefix = self.prefix

        n_masks = self.n_masks
        shapes = self.shapes

        if file_path is None:
            cmbs = self.cmbs_test
        else:
            cmbs = np.load(file_path)
        if nfl!=0:
            cmbs = cmbs+nfl*cmbs**2
        cmbs = cmbs-cmbs.min()
        cmbs = cmbs/cmbs.max()

        cmbs_pp = []
        for i in range(12):
             cmbs_pp.append(blocker(cmbs[i], ll, ll))

        cmbs_pp = np.concatenate(cmbs_pp)

        imasks = []
        masked_cmb = []

        for i in range(cmbs_pp.shape[0]):
            pcmb_cut = cmbs_pp[i]
            imask = mask_maker(shapes = shapes,
                                output_shape = (ll,ll),
                                n_masks = n_masks)

            masked_cmb.append((1-imask)*pcmb_cut)
            imasks.append(imask)

        imasks = np.array(imasks)[:,:,:,None]
        masked_cmb = np.array(masked_cmb)[:,:,:,None]

        gen_whole = self.generator.predict([masked_cmb,imasks])

        cmbs3 = np.zeros((12, 2048, 2048))

        np.save(prefix+'cmb_masked'+postfix,masked_cmb)
        np.save(prefix+'cmb_truth'+postfix,cmbs_pp)
        np.save(prefix+'cmb_gen'+postfix,gen_whole)

        for k in range(12):
            cmbs3[k] = deblock(gen_whole[k*l*l:(k+1)*l*l,:,:,0])

#        cl_path = prefix+'cl'+postfix+'.npy'
        if not os.path.exists(cl_path):
            fullsky = patch2sky(cmbs)
            fullsky = hp.pixelfunc.reorder(fullsky, inp=None, out=None, r2n=None, n2r=1)
            cl = hp.anafast(fullsky)
            np.save(cl_path[:-4],cl)
        else:
            cl = np.load(cl_path)        
        
        if not os.path.exists(clw_path):
            mmcmb = np.mean(cmbs)
            masked_cmb[masked_cmb==0] = mmcmb
            cmbsw = np.zeros((12, 2048, 2048))
            for k in range(12):
                cmbsw[k] = deblock(masked_cmb[k*l*l:(k+1)*l*l,:,:,0])
            fullskyw = patch2sky(cmbsw)
            fullskyw = hp.pixelfunc.reorder(fullskyw, inp=None, out=None, r2n=None, n2r=1)
            clw = hp.anafast(fullskyw)
            np.save(clw_path[:-4],clw)
        else:
            clw = np.load(clw_path)

        fullsky3 = patch2sky(cmbs3)
        fullsky3 = hp.pixelfunc.reorder(fullsky3, inp=None, out=None, r2n=None, n2r=1)
        cl3 = hp.anafast(fullsky3)
        np.save(prefix+'cl3'+postfix,cl3)

        ell = np.arange(len(cl))
        dl = ell * (ell + 1) * cl
        dl3 = ell * (ell + 1) * cl3
        dlw = ell * (ell + 1) * clw

        clw = np.load(clw_path)
        dlw = ell * (ell + 1) * clw
        
        xi2pred = np.sum((dl-dl3)**2)
        xi2w = np.sum((dl-dlw)**2)
        print(xi2pred/xi2w)

        
        plt.figure(figsize=(8, 5))
        plt.plot(ell, dl,color='b',ls='none',marker='.',alpha=0.2)
        plt.plot(ell, dl3,color='r',ls='none',marker='.',alpha=0.2)
        plt.plot(ell, dlw,color='g',ls='none',marker='.',alpha=0.2)
        plt.plot(ell, gaussian_filter(dl,10),color='b')
        plt.plot(ell, gaussian_filter(dl3,10),color='r')

        plt.xlabel("$\ell$",fontsize=20)
        plt.ylabel("$\ell(\ell+1)C_{\ell} [K]$",fontsize=15)
        plt.xscale('log')
        plt.xlim(1,2500)
        plt.grid()
        plt.subplots_adjust(left=0.12,bottom=0.13,right=0.99,top=0.99,wspace=0.01,hspace=0.01)
        plt.savefig(prefix+'power'+postfix+'.jpg',dpi=150)
        plt.close()

#        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

#        plt.sca(ax1)
#        hp.mollview(fullsky,cmap='jet',title='Truth',hold=1)
#        plt.sca(ax2)
#        hp.mollview(fullsky3,cmap='jet',title='Inpainted',hold=1)
#        plt.subplots_adjust(left=0.01,bottom=0.05,right=0.99,top=0.99,wspace=0.01,hspace=0.01)
#        plt.savefig(prefix+'maps'+postfix+'.jpg',dpi=150)
#        plt.close()

        return xi2pred/xi2w











def ch_mkdir(directory):
    """
    ch_mkdir : This function creates a directory if it does not exist.

    Arguments:
        directory (string): Path to the directory.

    --------
    Returns:
        null.		
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except:
            print('could not make the directory!')






