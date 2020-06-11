import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data
import random
from scipy.io import savemat 
import os
from sklearn.preprocessing import normalize


# In[2]:


def make_dataset(root):
    dataset = []
    with open(root) as f:
        #print(root)
        content = f.read().splitlines()
    for line in content:
        #path = np.load(line)
        dataset.append(line)
    return dataset
class lateSpeech(data.Dataset):
    def __init__(self,root1,root2):
        self.data1 = make_dataset(root1)
        #print(self.data1[0])
        self.data2 = make_dataset(root2)

        self.root1 = root1
        self.root2 = root2


    def __getitem__(self, index):
        
        Xfile1 = self.data1[index]

        de_file = self.data2[index]
        X1 = np.load(Xfile1)
        #X1 = normalize(X1)
        de = np.load(de_file)
        

        return torch.from_numpy(X1).t().float(),torch.from_numpy(de).float() 

    def __len__(self):
        return len(self.data1)


# In[3]:


def InputSize(x):
    a = np.load(x)
    b = a.shape
    return b[0]
def seqLen(x):
    a = np.load(x)
    b = a.shape
    return b[1]


# In[4]:
wlen = 480
nfft = 512
overlap = 360

input_file = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_train/train_wlen_'+ '%d'%wlen +'_nfft_'+'%d'%nfft+'_overlap_'+'%d'%overlap+'/reverb_rir5000_roomNum10_t600.9_loc500_8kHz.npy'
new_file = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb_valid/valid_wlen_'+'%d'%wlen+'_nfft_'+'%d'%nfft+'_overlap_'+'%d'%overlap+'/reverb_rir0500_roomNum10_t600.9_loc500_8kHz.npy'
print(seqLen(new_file),seqLen(input_file))

class param:
    #img_size = (80, 80)
    bs = 20
    lr = 10e-4
    epochs = 100
    hsize = 513
    hlayer = 6
    osize = 1026
    lstm_s = InputSize(input_file)
    lstm_l = seqLen(input_file)
    ts = 1

mix1_train = '/data/liyuy/PROJECTS/DEREVERB3/block_conv/stft_address/train/train_wlen_'+'%d'%wlen+'_nfft_'+'%d'%nfft+'_overlap_'+'%d'%overlap+'.txt'

de_train = "/data/liyuy/PROJECTS/DEREVERB3/block_conv/target_address/train/train_target.txt"

mix1_val = '/data/liyuy/PROJECTS/DEREVERB3/block_conv/stft_address/valid/valid_wlen_'+'%d'%wlen+'_nfft_'+'%d'%nfft+'_overlap_'+'%d'%overlap+'.txt'

de_val = "/data/liyuy/PROJECTS/DEREVERB3/block_conv/target_address/valid/valid_target.txt"

#mix1_test = "/data/liyuy/PROJECTS/DEREVERB3/block_conv/stft_address/test/test1/test_wlen_80_nfft_128_overlap_10.txt""

#de_test = "/data/liyuy/PROJECTS/DEREVERB3/block_conv/address_seg/target/testing/complex/multi/reverb_900_1_0.9.txt"


train_dl = data.DataLoader(lateSpeech(mix1_train,de_train),
                        batch_size=param.bs,
                        shuffle=True,
                        pin_memory=torch.cuda.is_available())
val_dl = data.DataLoader(lateSpeech(mix1_val,de_val),
                    batch_size=param.bs,
                    shuffle=False,
                    pin_memory=torch.cuda.is_available())

#test_dl = data.DataLoader(lateSpeech(mix1_test,de_test),
#                         batch_size=param.bs,
#                         shuffle=False)


# In[10]:


class LSTM_hn(nn.Module):
    def __init__(self):
        super(LSTM_hn,self).__init__()
        self.hsize = param.hsize
        self.hlayer = param.hlayer
        self.batchSize = param.bs
        self.h0 = self.init_hidden(self.hsize,self.hlayer)
        self.c0 = self.init_cell(self.hsize,self.hlayer)
        self.lstm = nn.LSTM(param.lstm_s,self.hsize,self.hlayer,batch_first=True) 
        self.fc1 = nn.Linear(param.lstm_l*self.hsize,self.hsize)
        self.fc2 = nn.Linear(self.hsize,param.osize)
    def init_hidden(self,hidden_size,hidden_layer):
        return Variable(torch.zeros(hidden_layer,self.batchSize, hidden_size).cuda())
    
    def init_cell(self,hidden_size,hidden_layer):
        return Variable(torch.zeros(hidden_layer,self.batchSize, hidden_size).cuda())   
        
    def forward(self,sig1):
        
        hx = self.h0
        cx = self.c0
        
        out,(hx,cx) = self.lstm(sig1,(hx,cx))
        
        new_out = out.contiguous().view(-1,param.lstm_l * self.hsize)
        
        output1 = Func.relu(self.fc1(new_out))
        
        de_out = Func.leaky_relu(self.fc2(output1))
        
        
        return de_out


# In[11]:

def transfer_to_time(output,batch_size):
    new_out = np.zeros((batch_size,1024))
    for i in range (0,batch_size):
        de_com = output[i].cpu().data.numpy()
        de_com = np.squeeze(de_com)
        de_real = de_com[0:513]
        de_real = np.expand_dims(de_real,axis=1)
      
   
        de_imag = de_com[513:]
        de_imag = np.expand_dims(de_imag,axis=1)
            
            
        fft_de = de_real + 1j * de_imag
        #print(fft_de.shape)   
        com_de = fft_de[1:512]
       
            #flip and take the complex conjugate part, and combine them together
        com_de = np.conj(np.flip(com_de))
        new_fft_de = np.concatenate((fft_de[0:513],com_de),axis=0)
        
        
        de = np.real(np.fft.ifft(new_fft_de,n=1024,axis=0))
        new_out[i,:] = np.squeeze(de)
    return torch.from_numpy(new_out).float()

model = LSTM_hn().cuda()
def weights(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data) 
model.apply(weights)
optim = torch.optim.Adam(model.parameters(), lr=param.lr)

criterion = nn.MSELoss()


# In[12]:


def get_loss(dl, model):
    loss = 0
    for X1, y1 in dl:
        X1, y1 = Variable(X1).cuda(), Variable(y1).cuda()
        output = model(X1)
        new_out = transfer_to_time(output,param.bs)
        new_de = transfer_to_time(y1,param.bs)
        ploss1 = criterion(output,y1)
        ploss2 = criterion(new_out,new_de)
        loss1 = (ploss1 + 3*ploss2)/4
        loss += loss1.cpu().item() * param.bs
    loss = loss / (len(val_dl.dataset))
    return loss


# In[13]:


iters = []
train_losses = []
val_losses = []

it = 0
min_loss = np.inf
bst_model_fpath = '/data/liyuy/PROJECTS/DEREVERB3/LSTM/exp5/model/bst_model_wlen_'+'%d'%wlen+'_nfft_'+'%d'%nfft+'_overlap_'+'%d'%overlap+'time_lambda4.pth'
model.train(True)

for epoch in range(1,param.epochs):
    loss = 0.0
    model.train(True)
    with torch.set_grad_enabled(True):
        for mag1,de_gtruth in train_dl:
            #print(mag.shape)
            mag1 = Variable(mag1.cuda())  # [N, 1, H, W]
            de_gtruth = Variable(de_gtruth.cuda())

            output = model(mag1)# [N, 2, H, W]
            new_out = transfer_to_time(output,param.bs)
            new_de = transfer_to_time(de_gtruth,param.bs)

            pLoss1 = criterion(output,de_gtruth)
            pLoss2 = criterion(new_out,new_de)
            pLoss = (pLoss1 + 3*pLoss2)/4
            

            pLoss = criterion(output,de_gtruth)

            loss += pLoss.cpu().item() * param.bs
            optim.zero_grad()
            pLoss.backward()
            optim.step()
        avgLoss = loss/len(train_dl.dataset)
      
    model.eval()
    
    val_loss = get_loss(val_dl, model)
    
     
    if val_loss < min_loss:
        min_loss = val_loss
        torch.save(model.state_dict(), bst_model_fpath)              
        print('Epoch {:2}, Train Loss:{:>.9f}, Validation Loss:{:>.9f}'.format(epoch,avgLoss,min_loss))
    print(epoch)




