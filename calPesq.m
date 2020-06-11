clear
clear all;

addpath('/home/liyuy/PROJECTS/other/PESQ');
test_set = 5;
wlen = 480;
nfft = 512;
overlap = 360; 
%path1 = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb/test5/';
out_in = '/data/liyuy/PROJECTS/DEREVERB3/LSTM/exp5/output/output_wav/test';
path1 = sprintf('%s%d_wlen_%d_nfft_%d_overlap_%dtime_lambda3_1024_',out_in,test_set,wlen,nfft,overlap);
%path1 = '/data/liyuy/PROJECTS/DEREVERB3/LSTM/exp5/output/output_wav/test5_wlen_160_nfft_256_overlap_120_1024_';

path_in = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/exp5/gr_truth/test';
path2 = sprintf('%s%d/',path_in,test_set);


T60s = [0.3 0.6 0.9];
i_loc = 1;
Fs = 8e3;
count = 1;
%roomNum = 1;
index = 1;
for j = 1:3
    T60 = T60s(j);
    est_path = sprintf('%st60%g/',path1,T60);
    scorelist = [];
    if test_set == 6 
        roomNum = 11;
    
    else
        roomNum = 1;
    end
    for i = 1:500
        if test_set == 5
            if i ~= 1 && mod((i -1),50) == 0
                roomNum = roomNum + 1;
            end
        else
           if i ~= 1 && mod((i-1),125) == 0
               roomNum = roomNum + 1;
           end
        end
        
        %for index = 1:50
        %est_file    = sprintf('%sreverb_rir%04d_roomNum%d_t60%g_loc%d_%dkHz.wav',path1,i,roomNum,T60,i,Fs/1000);
        est_file = sprintf('%srecons%04d_roomNum%d_t60%g_loc%d.wav',est_path,i,roomNum,T60,i);
            %est_file = sprintf('%sreverb_rir%04d_roomNum%d_t60%g_loc%d_%dkHz.wav',path1,i,roomNum,T60,i,Fs/1000);
        gr_file  = sprintf('%sreverb_rir%04d_roomNum%d_t60%g_loc%d_%dkHz.wav',path2,i,roomNum,T60,i,Fs/1000);
        if exist(est_file,'file')
            [y, Fs1] = audioread(est_file); % Loading Predicted/Enhanced signal            
            [speech, Fs2] = audioread(gr_file); % Loading clean speech signal
        end 
        if Fs1 ~= Fs
            
            y = resample(y,Fs,Fs1);
        elseif Fs2 ~= Fs
            speech = resample(speech,Fs,Fs2);
        end
            len1 = length(y);
            len2 = length(speech);
            if len1 < len2
                speech = speech(1:len1);
            else
                y = y(1:len2);
            end
            score = pesqscore(speech,y,Fs);
            %disp(score);
            scorelist(i)=score;
            
        
    end
    ave = sum(scorelist)/length(scorelist);
    stdn = std(scorelist);
    disp(ave);
    disp(stdn);
end

