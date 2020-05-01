bst_model_fpath = '/data/liyuy/PROJECTS/DEREVERB3/LSTM/exp5/model/bst_model_wlen_80_nfft_128_overlap_20.pth'


outputPath = '/data/liyuy/PROJECTS/DEREVERB3/LSTM/exp5/output/output_de_all/test5_wlen_80_nfft_128_overlap_20/'
if path.exists(outputPath) == False:
    os.mkdir(outputPath)
model.load_state_dict(torch.load(bst_model_fpath))


model.eval()
#new_count = 1
index = 1
Fs = 8e3
loss = 0.0
t60s = [0.3,0.6,0.9]
t60_i = 0   
i_loc = 1
new_i = 1
fft_length = 8000

with torch.set_grad_enabled(False):
    for mag1,de_gtruth in test_dl:
            #print(mag.shape)
        mag1 = Variable(mag1.cuda())  # [N, 1, H, W]
        
        de_gtruth = Variable(de_gtruth.cuda())
        
            # [N, H, W] with class indices (0, 1)
        output1 = model(mag1)# [N, 2, H, W]
        
        pLoss = criterion(output1,de_gtruth)
        
        loss += pLoss.cpu().item() * param.bs
        avgLoss = loss/len(test_dl.dataset)
        
 
       
        if index != 1 and (index - 1) % 3 == 0:
            new_i += 1
            t60_i = 0
            i_loc += 1
        de_com = output1.cpu().data.numpy()
        de_com = np.squeeze(de_com)
        #print(de_com.shape)   
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
        
        de = np.real(np.fft.ifft(new_fft_de,n=fft_length,axis=0))
         
       
        t60 = t60s[t60_i]
        outPath = outputPath +'te_de_'+ "{0:0=4d}".format(new_i) + '_t60' + '%.1f' % t60 + '_loc' + '%d' % (i_loc) +'recons.mat'
        savemat(outPath,{'de':de})
            
        index += 1
        #i_loc += 1
        t60_i += 1
    print('Test Loss:{:>.9f}'.format(avgLoss))    
