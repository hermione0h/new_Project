clear all;
wlen = 480;
nfft = 512;
overlap = 360;
test_set = 5;
rir_in = '/data/liyuy/PROJECTS/DEREVERB3/LSTM/exp5/output/output_de_all/test';
rir_path = sprintf('%s%d_wlen_%d_nfft_%d_overlap_%dtime_lambda3_1024/te_de_',rir_in,test_set,wlen,nfft,overlap);

clean_path = '/data/SpeechCorpora/TIMIT/test_corpus/corpus/';
out_in = '/data/liyuy/PROJECTS/DEREVERB3/LSTM/exp5/output/output_wav/test';
out_path = sprintf('%s%d_wlen_%d_nfft_%d_overlap_%dtime_lambda3_1024_',out_in,test_set,wlen,nfft,overlap);



i_loc = 1;
Fs = 8e3;
t60s = [0.3 0.6 0.9];
len = 45000;
%roomNum = 11;
index = 1;
for j = 1:3
    n_t60 = t60s(j);
        out1 = sprintf('%st60%g/',out_path,n_t60);
        if ~exist(out1,'dir')
            mkdir(out1);
        end
        
    if test_set == 6 
        roomNum = 11;
    
    else
        roomNum = 1;
    end
    index = 1;
    for i = 1:500
    
        

       if test_set == 5
            if i ~= 1 && mod((i-1),50) == 0
                roomNum = roomNum +1;
            end
       else
           if i ~= 1 && mod((i-1),125) == 0
               roomNum = roomNum + 1;
               index = 1;
           end
       end
    disp(index);
    cs_count = index+880;
    cs_path = sprintf('%s%d.wav',clean_path,cs_count);
    %for index = 1:50
    rir_out = sprintf('%s%04d_t60%g_loc%drecons.mat',rir_path,i,n_t60,i);
    [cs_data,Fs1] = audioread(cs_path);
        if Fs1 ~= Fs
            cs_data = resample(cs_data,Fs,Fs1);
        end
        newLen = length(cs_data);
            if newLen > len
                cs_data = cs_data(1:len);
            elseif newLen < len
                padding = zeros((len-newLen),1);
                cs_data = vertcat(cs_data,padding);
            end
    rir = load(rir_out);
    de = rir.de;
    disp(length(de));
    new_rec = conv(cs_data,de);
    disp(length(new_rec))
    %n_t60 = 0.9;
    save_file = sprintf('%srecons%04d_roomNum%d_t60%g_loc%d.wav',out1,i,roomNum,n_t60,i);
    audiowrite(save_file,new_rec,Fs);
    index = index+1;
    end 
     
 end
    
    
    
