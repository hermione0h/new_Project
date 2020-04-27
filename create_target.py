#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat
import numpy as np
from os import path
import os


# In[2]:



roomNum = 1 # starting room 1 and total 10 rooms in train
t60s = [0.3,0.6,0.9]
loc = 1
Fs = 8000
num_samples = 5000 # total number of original clean signals for training set
num_rir_fft = 513 # number of samples in target fft


# In[3]:



# concatenate imaginary and real parts into vector, the first # 513 samples are real, the second # 513 samples are imaginary
input_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/train_de_complex/reverb_rir'
out_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/train_de_ir/'  
if path.exists(out_path) == False:
    os.mkdir(out_path)
for i in range(1,num_samples+1):
    if i != 1 and (i - 1) % 500 == 0:
        roomNum += 1
        loc = 1
    for x in range(0,len(t60s)):
        t60 = t60s[x]
        new_comb = np.zeros((1,num_rir_fft * 2))
        new_path = input_path + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % loc + '_' + '%d' % (Fs/1000) + 'kHz.mat'
        a = loadmat(new_path)
        newFFT = a['de_fft']

        real = newFFT.real
        imag = newFFT.imag

        new_comb[:,0:num_rir_fft] = real
        new_comb[:,num_rir_fft:] = imag
        new_comb = np.squeeze(new_comb)
        new_out = out_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
        np.save(new_out,new_comb)
    loc += 1


# In[4]:


roomNum = 1 # start room 1 and total 10 rooms in validation
loc = 1
num_samples = 500 # total number of clean signals in validation set


# In[5]:


# generate validation labels 
input_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/valid_de_complex/reverb_rir'
out_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/valid_de_ir/'
if path.exists(out_path) == False:
    os.mkdir(out_path)
    
for i in range(1,num_samples+1):
    if i != 1 and (i - 1) % 50 == 0:
        roomNum += 1
        
    for x in range(0,len(t60s)):
        t60 = t60s[x]
    
        new_comb = np.zeros((1,num_rir_fft*2))
        new_path = input_path + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % loc + '_' + '%d' % (Fs/1000) + 'kHz.mat'
        a = loadmat(new_path)
        newFFT = a['de_fft']
            
        real = newFFT.real
        imag = newFFT.imag
    
        new_comb[:,0:num_rir_fft] = real
        new_comb[:,num_rir_fft:] = imag
        new_comb = np.squeeze(new_comb)
        new_out = out_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
        np.save(new_out,new_comb)
    loc += 1
        


# In[6]:


roomNums = [11,12,13,14] #unseen rir room number
num_samples = 500
num_rooms = 4


# In[7]:


# generate first 4 test labels with 4 unseen rooms
input_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/test'
out_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/test'
for a in range(0,num_rooms):
    roomNum = roomNums[a]
    new_in = input_path + '%d'%(a+1) + '_de_complex/reverb_rir'
    new_out = out_path + '%d'%(a+1) + '_de_ir/'
    if path.exists(new_out) == False:
        os.mkdir(new_out)
    loc = 1
    for i in range(1,num_samples+1):

        for x in range(0,len(t60s)):
            t60 = t60s[x]

            new_comb = np.zeros((1,num_rir_fft*2))
            new_path = new_in + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % loc + '_' + '%d' % (Fs/1000) + 'kHz.mat'
            a = loadmat(new_path)
            newFFT = a['de_fft']

            real = newFFT.real
            imag = newFFT.imag

            new_comb[:,0:num_rir_fft] = real
            new_comb[:,num_rir_fft:] = imag
            new_comb = np.squeeze(new_comb)
            new_out_path = new_out + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
            np.save(new_out_path,new_comb)
        loc += 1


# In[8]:


roomNum = 1
loc = 1
num_samples = 500


# In[9]:


# generate test5 labels from seen room 1-10
input_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/test5_de_complex/reverb_rir'
out_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/test5_de_ir/'

if path.exists(out_path) == False:
    os.mkdir(out_path)

for i in range(1,num_samples+1):
    if i != 1 and (i - 1) % 50 == 0:
        roomNum += 1
    for x in range(0,len(t60s)):
        t60 = t60s[x]

        new_comb = np.zeros((1,num_rir_fft*2))
        new_path = input_path + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % loc + '_' + '%d' % (Fs/1000) + 'kHz.mat'
        a = loadmat(new_path)
        newFFT = a['de_fft']

        real = newFFT.real
        imag = newFFT.imag

        new_comb[:,0:num_rir_fft] = real
        new_comb[:,num_rir_fft:] = imag
        new_comb = np.squeeze(new_comb)
        new_out_path = out_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
        np.save(new_out_path,new_comb)
    loc += 1


# In[ ]:




