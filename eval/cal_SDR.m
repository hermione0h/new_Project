clear
clear all;

addpath('/home/liyuy/PROJECTS/other/eval');

clean_path = '/data/liyuy/PROJECTS/DEREVERB2/DNN/exp1/ground_truth_de_wav/2to1/';
mix_path = '/data/liyuy/PROJECTS/DEREVERB2/input/test/2to1/';
%est_path = '/data/liyuy/PROJECTS/DEREVERB2/wpe/output/test/2to1/';
%est_path = '/data/liyuy/PROJECTS/DEREVERB2/DNN/exp1/output_joint/de_wav_0.91/';
T60 = 0.9;
i_loc = 1;
Fs = 16e3;
index = 1;
roomNum = 6;
for i = 1:1000
        
        %est_file    = sprintf('%sreverb_rir%04d_roomNum%d_t60%g_loc%d_%dkHz.wav',est_path,i,6,T60,i_loc,Fs/1000);
        %est_file = sprintf('%srecons%04d_roomNum%d_t60%g_loc%d.wav',est_path,i,6,T60,i_loc);
        
        gr_file  = sprintf('%sreverb_rir%04d_roomNum%d_t60%g_loc%d_%dkHz.wav',clean_path,i,6,T60,i_loc,Fs/1000);
        mix_file  = sprintf('%sreverb_rir%04d_roomNum%d_t60%g_loc%d_%dkHz.wav',mix_path,i,6,T60,i_loc,Fs/1000);
        if exist(mix_file,'file')
            %[y, Fs1] = audioread(est_file); % Loading Predicted/Enhanced signal            
            [clean, Fs2] = audioread(gr_file); % Loading clean speech signal
            [mix,Fs3] = audioread(mix_file);
            
        end 
        
        len1 = length(clean);
        len2 = length(mix);
        if len1 < len2
            mix = mix(1:len1);
            
        else
            clean = clean(1:len1);
        end
        N = mix-clean;
        %score = pesqscore(y/1000,speech);
        %score = stoi(y,speech,Fs1);
        [s_t,e_i,e_a] = bss_decomp_gain(mix',1,clean',N');
        [SDR,SIR,SAR] = bss_crit(s_t,e_i,e_a);
        %disp(SDR);
        scorelist(index)=SDR;
        index = index + 1;
        if mod(i,2) == 0
           i_loc = i_loc + 1; 
        end
end

ave = sum(scorelist)/1000;
stdn = std(scorelist);
disp(ave);
disp(stdn);
