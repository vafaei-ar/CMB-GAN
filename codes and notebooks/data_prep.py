import os
import sys
import numpy as np
import healpy as hp
from ccgpack import ch_mkdir
from ccgpack import download   
from ccgpack import sky2patch, patch2sky
import matplotlib.pylab as plt

def report(count, blockSize, totalSize):
  	percent = int(count*blockSize*TOTALSIZE)
  	sys.stdout.write("\r%d%%" % percent + ' complete')
  	sys.stdout.flush() 
  
def download_simulation(i,typ,com_s):
    name_out = './maps/dx12_v3_{}_{}_mc_{:05d}_raw.fits'.format(com_s,typ,i)
    
    if os.path.exists(name_out):
        x = os.path.getsize(name_out)
        if x!=603987840:
            print(name_out,'is corrupted!')
            os.remove(name_out)
    
    if not os.path.exists(name_out):
        ch_mkdir('maps')
        global TOTALSIZE
        TOTALSIZE = 100/(589836*1024)
        name_in = 'http://pla.esac.esa.int/pla/aio/product-action?SIMULATED_MAP.FILE_ID=dx12_v3_{}_{}_mc_{:05d}_raw.fits'.format(com_s,typ,i)
        download(name_in,name_out,report=report)
    else:
        print(name_out,'exist!')
    return name_out

# com_s can be ['smica','commander','sevem','nilc']:
# i can be 0 to 999
com_s = 'smica'
typ = 'cmb'

#for i_sim in range(11,15):
#    name_out = download_simulation(i_sim,typ,com_s)

#ch_mkdir('patches')

#By convention 0 is temperature, 1 is Q, 2 is U
for i in range(0,15):
    m = hp.read_map('./maps/dx12_v3_smica_cmb_mc_{:05d}_raw.fits'.format(i),field=2,nest=1)
    patches = sky2patch(m)

    np.save('./patches/U'+str(i),patches)


#mask_name = 'COM_Mask-Int_2048_R3.00.fits'
#dg_mask_name = 'COM_Mask-Int_2048_R3.00_dg.fits'
#    
#if not os.path.exists(mask_name):
#    global TOTALSIZE
#    TOTALSIZE = 100/(192.0*1024*1024)
#    download('http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits',
#             mask_name,
#             report=report)  

#mask = hp.read_map('COM_Mask-Pol_2048_R3.00.fits',nest=1)
#patches = sky2patch(mask)
#np.save('mask_pol',patches)
