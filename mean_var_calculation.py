#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
from os import listdir
from os import path
from os.path import isfile, join
import os
import numpy as np


# In[2]:


Fs = 8e3
nfft = 2 ** 7
wlen_time = [10,20,40,60,80]
overlap_perc = [0.125,0.25,0.5,0.75]

t60s = [0.3,0.6,0.9]
num_samples = 5000
num_rirs = 500


# In[ ]:


input_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_train/train_wlen_'
out_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/mean_val_file/'
if path.exists(out_path) == False:
            os.mkdir(out_path)

for a in range(0,len(wlen_time)):
    wlen = int(wlen_time[a] * 1e-3 * Fs)
    if wlen > nfft:
        nfft = nfft * 2
    for b in range(0,len(overlap_perc)):
        overlap = int(overlap_perc[b] * wlen)
        hop_size = wlen - overlap
        newin_path = input_path + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '/'
        newout_file = out_path + 'train_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '.txt'
        new_comb = None
       
        for f in listdir(newin_path):
            new_path = join(newin_path,f)
            comb_input = np.load(new_path)
            
            if type(new_comb) == type(None):
                new_comb = comb_input
            else:
                
                new_comb = np.concatenate((new_comb,comb_input),axis=1)
            #print(new_comb.shape)
            
        mean_comb = np.mean(new_comb)
        std_comb = np.std(new_comb)
        print(mean_comb,std_comb)
            
        file1 = open(newout_file,"a")
         
        file1.write('%.6f'% mean_comb+ ', ' + '%.6f'%std_comb )
        file1.close()
        


# In[ ]:




