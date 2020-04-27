#!/usr/bin/env python
# coding: utf-8

# In[12]:


from os import path
import os


# In[1]:


Fs = 8e3
nfft = 2 ** 7 # recomended to be power of 2 and greater than win_len, this is the smallest that meet 10 millisec
wlen_time = [10,20,40,60,80] # millisec
overlap_perc = [0.125,0.25,0.5,0.75] # percentage of win_len

t60s = [0.3,0.6,0.9]
num_samples = 5000
num_rirs = 500


# In[4]:


# generate TXT file for training input

trainPyPath= '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_train/'
txtPyPath = '/data/liyuy/PROJECTS/DEREVERB3/block_conv/stft_address/train/'
for a in range(0,len(wlen_time)):
    wlen = int(wlen_time[a] * 1e-3 * Fs)
    if wlen > nfft: # To meet the condition, we multiply by 2 when nfft < wlen
        nfft = nfft * 2
    for b in range(0,len(overlap_perc)):
        overlap  = int(overlap_perc[b] * wlen)
        
        train_path = trainPyPath + 'train_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '/' 
        save_file = txtPyPath + 'train_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '.txt' 
        file1 = open(save_file,"a")
        roomNum = 1
        i_loc = 1
        for i in range(1,num_samples+1):
            if i != 1 and (i - 1) % num_rirs == 0:
                roomNum += 1
                i_loc = 1
            for x in range(0,len(t60s)):
                t60 = t60s[x]
                Xfile = train_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
                file1.write(Xfile+"\n")
            i_loc += 1
        file1.close()


# In[17]:


# generate txt file for train target
Fs = 8e3

t60s = [0.3,0.6,0.9]
num_samples = 5000
num_rirs = 500
roomNum = 1
i_loc = 1

trainPyPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/train_de_ir/'
txtPyPath = '/data/liyuy/PROJECTS/DEREVERB3/block_conv/target_address/train/'

save_file = txtPyPath + 'train_target.txt'
file1 = open(save_file,"a")

for i in range(1,num_samples+1):
    if i != 1 and (i - 1) % num_rirs == 0:
        roomNum += 1
        i_loc = 1
    for x in range(0,len(t60s)):
        t60 = t60s[x]
        Xfile = trainPyPath + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
        file1.write(Xfile+"\n")
    i_loc += 1
file1.close()

    

    


# In[5]:


Fs = 8e3
nfft = 2 ** 7
wlen_time = [10,20,40,60,80]
overlap_perc = [0.125,0.25,0.5,0.75]

t60s = [0.3,0.6,0.9]
num_samples = 500
num_rirs = 50


# In[6]:


# generate address file for validation input

validPyPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_valid/'
txtPyPath = '/data/liyuy/PROJECTS/DEREVERB3/block_conv/stft_address/valid/'
for a in range(0,len(wlen_time)):
    wlen = int(wlen_time[a] * 1e-3 * Fs)
    if wlen > nfft:
        nfft = nfft * 2
    for b in range(0,len(overlap_perc)):
        overlap = int(overlap_perc[b] * wlen)
        valid_path = validPyPath + 'valid_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '/'
        save_file = txtPyPath + 'valid_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '.txt' 
        file1 = open(save_file,"a")
        roomNum = 1
        i_loc = 1
        for i in range(1,num_samples+1):
            if i != 1 and (i - 1) % num_rirs == 0:
                roomNum += 1

            for x in range(0,len(t60s)):
                t60 = t60s[x]

                Xfile = valid_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
                file1.write(Xfile+"\n")
                
            i_loc += 1
    
        file1.close()


# In[18]:


# generate txt file for train target
Fs = 8e3

t60s = [0.3,0.6,0.9]
num_samples = 500
num_rirs = 50
roomNum = 1
i_loc = 1

validPyPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/target/all_t60/valid_de_ir/'
txtPyPath = '/data/liyuy/PROJECTS/DEREVERB3/block_conv/target_address/valid/'

save_file = txtPyPath + 'valid_target.txt'
file1 = open(save_file,"a")

for i in range(1,num_samples+1):
    if i != 1 and (i - 1) % num_rirs == 0:
        roomNum += 1
    for x in range(0,len(t60s)):
        t60 = t60s[x]
        Xfile = validPyPath + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
        file1.write(Xfile+"\n")
    i_loc += 1
file1.close()


# In[7]:


Fs = 8e3
nfft = 2 ** 7 # recomended to be power of 2 and greater than win_len, this is the smallest that meet 10 millisec
wlen_time = [10,20,40,60,80] # millisec
overlap_perc = [0.125,0.25,0.5,0.75] # percentage of win_len

t60s = [0.3,0.6,0.9]
num_samples = 500
num_rirs = 500
roomNums = [11,12,13,14] # 4 rooms with unseen RIRs


# In[13]:


# generate address file for unseen input

testPyPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_test'
txtPath = '/data/liyuy/PROJECTS/DEREVERB3/block_conv/stft_address/test/test'
for r in range(0,len(roomNums)):
    roomNum = roomNums[r]
    new_test = testPyPath + '%d'%(r+1) + '/'
    new_out = txtPath + '%d'%(r+1) + '/'
    if path.exists(new_out) == False:
        os.mkdir(new_out)
    for a in range(0,len(wlen_time)):
        wlen = int(wlen_time[a] * 1e-3 * Fs)
        if wlen > nfft:
            nfft = nfft * 2
        for b in range(0,len(overlap_perc)):
            overlap = int(overlap_perc[b] * wlen)
            
            test_path = new_test + 'test_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '/'
            save_file = new_out + 'test_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '.txt' 
            file1 = open(save_file,"a")
            i_loc = 1
            for i in range(1,num_samples+1):

                for x in range(0,len(t60s)):
                    t60 = t60s[x]
                    Xfile = test_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
                    file1.write(Xfile+"\n")
                i_loc += 1
            file1.close()


# In[14]:


Fs = 8e3
nfft = 2 ** 7 # recomended to be power of 2 and greater than win_len, this is the smallest that meet 10 millisec
wlen_time = [10,20,40,60,80] # millisec
overlap_perc = [0.125,0.25,0.5,0.75] # percentage of win_len

t60s = [0.3,0.6,0.9]
num_samples = 500
num_rirs = 50

roomNums = 1 # start room number, total 10 rooms (seen RIRs)


# In[15]:


testPyPath = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_test5/'
txtPyPath = '/data/liyuy/PROJECTS/DEREVERB3/block_conv/stft_address/test/test5/'
if path.exists(txtPyPath) == False:
        os.mkdir(txtPyPath)
for a in range(0,len(wlen_time)):
    wlen = int(wlen_time[a] * 1e-3 * Fs)
    if wlen > nfft:
        nfft = nfft * 2
    for b in range(0,len(overlap_perc)):
        overlap = int(overlap_perc[b] * wlen)
        test_path = testPyPath + 'test_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '/'
        save_file = txtPyPath + 'test_wlen_' + '%d' % wlen + '_nfft_' + '%d' % nfft + '_overlap_' + '%d' % overlap + '.txt' 
        file1 = open(save_file,"a")
        roomNum = 1
        i_loc = 1
        for i in range(1,num_samples+1):
            if i != 1 and (i - 1) % num_rirs == 0:
                roomNum += 1

            for x in range(0,len(t60s)):
                t60 = t60s[x]

                Xfile = test_path + 'reverb_rir' + "{0:0=4d}".format(i) + '_roomNum' + '%d' % roomNum + '_t60' + '%.1f' % t60 + '_loc' + '%d' % i_loc + '_' + '%d' % (Fs/1000) + 'kHz.npy'
                file1.write(Xfile+"\n")
                
            i_loc += 1
    
        file1.close()


# In[ ]:




