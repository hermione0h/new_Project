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


def cal_mean(dir_in,nfft,hop_size,wlen):
    for f in listdir(dir_in):
        sum_col = 0
        new_path = join(dir_in,f)
        x,fs = librosa.load(new_path,sr=None)
                
        reverb_stft = librosa.stft(x,n_fft = nfft,hop_length = hop_size,win_length = wlen)
        dims = reverb_stft.shape
        row_n = dims[0]
        col_n = dims[1]

        new_comb = np.zeros((dim_size*row_n,col_n))# create a numpy array of size [dim_size * row_n,col_n]
        sum_m = np.zeros((dim_size*row_n,))
        # This way can make sure three features can pass in right time sequence
        phase = np.angle(reverb_stft)
        cos_stft = np.cos(phase)
        sin_stft = np.sin(phase)
        logmag_stft = np.log(np.absolute(reverb_stft)+np.finfo(float).eps)
        new_comb[0:row_n,:] = logmag_stft
        new_comb[row_n:2*row_n,:] = sin_stft
        new_comb[2*row_n:,:] = cos_stft
        sum_m += np.sum(new_comb,axis=1)
        sum_col += col_n
            
    mean_comb = sum_m/col_n
    return mean_comb
                


# In[7]:


def cal_var(dir_in,mean_c,nfft,hop_size,wlen):
    for f in listdir(dir_in):
        sum_col = 0
        new_path = join(dir_in,f)
        x,fs = librosa.load(new_path,sr=None)
                
        reverb_stft = librosa.stft(x,n_fft = nfft,hop_length = hop_size,win_length = wlen)
        dims = reverb_stft.shape
        row_n = dims[0]
        col_n = dims[1]

        new_comb = np.zeros((dim_size*row_n,col_n))# create a numpy array of size [dim_size * row_n,col_n]
        sum_m = np.zeros((dim_size*row_n,))
        # This way can make sure three features can pass in right time sequence
        phase = np.angle(reverb_stft)
        cos_stft = np.cos(phase)
        sin_stft = np.sin(phase)
        logmag_stft = np.log(np.absolute(reverb_stft)+np.finfo(float).eps)
        new_comb[0:row_n,:] = logmag_stft
        new_comb[row_n:2*row_n,:] = sin_stft
        new_comb[2*row_n:,:] = cos_stft
        new_input = np.square((new_comb.transpose()-mean_c).transpose())
        #print(new_input.shape)
        sum_m += np.sum(new_input,axis=1)
        sum_col += col_n
        
    var_sq = sum_m / sum_col
    var = np.sqrt(var_sq)
    return var
                


# In[11]:


Fs = 8e3
nfft = 2 ** 7 # recomended to be power of 2 and greater than win_len, this is the smallest that meet 10 millisec
wlen_time = [10,20,40,60,80] # millisec
overlap_perc = [0.125,0.25,0.5,0.75] # percentage of win_len

t60s = [0.3,0.6,0.9]
num_samples = 5000
num_rirs = 500
dim_size = 3


# In[12]:


trainPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb/train/'
outPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_train/'
'''
for a in range(0,len(wlen_time)):
    wlen = int(wlen_time[a] * 1e-3 * Fs)
    if wlen > nfft: # To meet the condition, we multiply by 2 when nfft < wlen
        nfft = nfft * 2
    for b in range(0,len(overlap_perc)):
        overlap = int(overlap_perc[b] * wlen)
        hop_size = wlen - overlap
        newout_path = outPath + 'train_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '/'
        if path.exists(newout_path) == False:
            os.mkdir(newout_path)
        roomNum = 1
        i_loc = 1
        mean_comb = cal_mean(trainPath,nfft,hop_size,wlen)
        var = cal_var(trainPath,mean_comb,nfft,hop_size,wlen)
        for i in range(1,num_samples+1):
            if i != 1 and (i - 1) % num_rirs == 0:
                roomNum += 1
                i_loc = 1
            for x in range(0,len(t60s)):
                t60 = t60s[x]

                input_path = trainPath + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_8kHz.wav'

                x,fs = librosa.load(input_path,sr=None)
                
                reverb_stft = librosa.stft(x,n_fft = nfft,hop_length = hop_size,win_length = wlen)
                dims = reverb_stft.shape
                row_n = dims[0]
                col_n = dims[1]
                
                new_comb = np.zeros((dim_size*row_n,col_n)) # create a numpy array of size [dim_size * row_n,col_n]
                # This way can make sure three features can pass in right time sequence
                phase = np.angle(reverb_stft)
                cos_stft = np.cos(phase)
                sin_stft = np.sin(phase)
                logmag_stft = np.log(np.absolute(reverb_stft)+np.finfo(float).eps)
                new_comb[0:row_n,:] = logmag_stft
                new_comb[row_n:2*row_n,:] = sin_stft
                new_comb[2*row_n:,:] = cos_stft
                out_comb = np.divide((new_comb.transpose() - mean_comb),(var+np.finfo(float).eps)).transpose()
                #print(out_comb.shape)
                output_path = newout_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_'+'%d'%(Fs/1000) +'kHz.npy'
                np.save(output_path,out_comb)
            i_loc += 1
    
        


# In[6]:


Fs = 8e3
nfft = 2 ** 7
wlen_time = [10,20,40,60,80]
overlap_perc = [0.125,0.25,0.5,0.75]

t60s = [0.3,0.6,0.9]
num_samples = 500
num_rirs = 50
dim_size = 3


# In[12]:


validPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb/valid/'
outPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_valid/'
for a in range(0,len(wlen_time)):
    wlen = int(wlen_time[a] * 1e-3 * Fs)
    if wlen > nfft:
        nfft = nfft * 2
    for b in range(0,len(overlap_perc)):
        overlap = int(overlap_perc[b] * wlen)
        hop_size = wlen - overlap
        newout_path = outPath + 'valid_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '/'
        if path.exists(newout_path) == False:
            os.mkdir(newout_path)
        roomNum = 1
        i_loc = 1
        mean_comb = cal_mean(trainPath,nfft,hop_size,wlen)
        var = cal_var(trainPath,mean_comb,nfft,hop_size,wlen)
        for i in range(1,num_samples+1):
            if i != 1 and (i - 1) % num_rirs == 0:
                roomNum += 1

            for x in range(0,len(t60s)):
                t60 = t60s[x]

                input_path = validPath + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_8kHz.wav'

                x,fs = librosa.load(input_path,sr=None)
                
                reverb_stft = librosa.stft(x,n_fft = nfft,hop_length = hop_size,win_length = wlen)
                dims = reverb_stft.shape
                
                row_n = dims[0]
                col_n = dims[1]
                
                new_comb = np.zeros((dim_size*row_n,col_n))
                phase = np.angle(reverb_stft)
                cos_stft = np.cos(phase)
                sin_stft = np.sin(phase)
                logmag_stft = np.log(np.absolute(reverb_stft)+np.finfo(float).eps)
                new_comb[0:row_n,:] = logmag_stft
                new_comb[row_n:2*row_n,:] = sin_stft
                new_comb[2*row_n:,:] = cos_stft
                out_comb = np.divide((new_comb.transpose() - mean_comb),(var+np.finfo(float).eps)).transpose()
                output_path = newout_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_8kHz.npy'
                np.save(output_path,out_comb)
            i_loc += 1
    
        

'''
# In[7]:


Fs = 8e3
nfft = 2 ** 7 # recomended to be power of 2 and greater than win_len, this is the smallest that meet 10 millisec
wlen_time = [10,20,40,60,80] # millisec
overlap_perc = [0.125,0.25,0.5,0.75] # percentage of win_len

t60s = [0.3,0.6,0.9]
num_samples = 500
num_rirs = 500
dim_size = 3
roomNums = [11,12,13,14] # 4 rooms with unseen RIRs


# In[8]:


testPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb/test'
outPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_test'
for r in range(0,len(roomNums)):
    roomNum = roomNums[r]
    new_test = testPath + '%d'%(r+1) + '/'
    new_out = outPath + '%d'%(r+1) + '/'
    if path.exists(new_out) == False:
        os.mkdir(new_out)
    nfft = 2 ** 7
    for a in range(0,len(wlen_time)):
        wlen = int(wlen_time[a] * 1e-3 * Fs)
        if wlen > nfft:
            nfft = nfft * 2
        for b in range(0,len(overlap_perc)):
            overlap = int(overlap_perc[b] * wlen)
            hop_size = wlen - overlap
            newout_path = new_out + 'test_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '/'
            if path.exists(newout_path) == False:
                os.mkdir(newout_path)
            
            i_loc = 1
            mean_comb = cal_mean(trainPath,nfft,hop_size,wlen)
            var = cal_var(trainPath,mean_comb,nfft,hop_size,wlen)
            for i in range(1,num_samples+1):

                for x in range(0,len(t60s)):
                    t60 = t60s[x]

                    input_path = new_test + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_8kHz.wav'

                    x,fs = librosa.load(input_path,sr=None)
                    reverb_stft = librosa.stft(x,n_fft = nfft,hop_length = hop_size,win_length = wlen)
                    dims = reverb_stft.shape
                    row_n = dims[0]
                    col_n = dims[1]
                    new_comb = np.zeros((dim_size*row_n,col_n))
                    phase = np.angle(reverb_stft)
                    cos_stft = np.cos(phase)
                    sin_stft = np.sin(phase)
                    logmag_stft = np.log(np.absolute(reverb_stft)+np.finfo(float).eps)
                    new_comb[0:row_n,:] = logmag_stft
                    new_comb[row_n:2*row_n,:] = sin_stft
                    new_comb[2*row_n:,:] = cos_stft
                    out_comb = np.divide((new_comb.transpose() - mean_comb),(var+np.finfo(float).eps)).transpose()
                    output_path = newout_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_8kHz.npy'
                    np.save(output_path,out_comb)
                i_loc += 1


# In[2]:

'''
Fs = 8e3
nfft = 2 ** 7 # recomended to be power of 2 and greater than win_len, this is the smallest that meet 10 millisec
wlen_time = [10,20,40,60,80] # millisec
overlap_perc = [0.125,0.25,0.5,0.75] # percentage of win_len

t60s = [0.3,0.6,0.9]
num_samples = 500
num_rirs = 50
dim_size = 3
roomNums = 1 # start room number, total 10 rooms (seen RIRs)


# In[4]:


testPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb/test5/'
outPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_test5/'
if path.exists(outPath) == False:
            os.mkdir(outPath)
for a in range(0,len(wlen_time)):
    wlen = int(wlen_time[a] * 1e-3 * Fs)
    if wlen > nfft:
        nfft = nfft * 2
    for b in range(0,len(overlap_perc)):
        overlap = int(overlap_perc[b] * wlen)
        hop_size = wlen - overlap
        newout_path = outPath + 'test_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '/'
        if path.exists(newout_path) == False:
            os.mkdir(newout_path)
        roomNum = 1
        i_loc = 1
        mean_comb = cal_mean(trainPath,nfft,hop_size,wlen)
        var = cal_var(trainPath,mean_comb,nfft,hop_size,wlen)
        for i in range(1,num_samples+1):
            if i != 1 and (i - 1) % num_rirs == 0:
                roomNum += 1

            for x in range(0,len(t60s)):
                t60 = t60s[x]

                input_path = testPath + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_8kHz.wav'

                x,fs = librosa.load(input_path,sr=None)
                reverb_stft = librosa.stft(x,n_fft = nfft,hop_length = hop_size,win_length = wlen)
                dims = reverb_stft.shape
                row_n = dims[0]
                col_n = dims[1]
                new_comb = np.zeros((dim_size*row_n,col_n))
                phase = np.angle(reverb_stft)
                cos_stft = np.cos(phase)
                sin_stft = np.sin(phase)
                logmag_stft = np.log(np.absolute(reverb_stft)+np.finfo(float).eps)
                new_comb[0:row_n,:] = logmag_stft
                new_comb[row_n:2*row_n,:] = sin_stft
                new_comb[2*row_n:,:] = cos_stft
                out_comb = np.divide((new_comb.transpose() - mean_comb),(var+np.finfo(float).eps)).transpose()
                output_path = newout_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_8kHz.npy'
                np.save(output_path,out_comb)
            i_loc += 1
    
'''        


# In[ ]:




