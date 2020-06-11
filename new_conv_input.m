clear all;
clear
rirPath = '/home/liyuy/PROJECTS/block_conv/room';

data_path = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/test/';

%out_path1 = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/input/all_t60/test1_fft_900/';
%out_path2 = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/input/all_t60/test2_cos_900/';
%out2 = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/input/all_t60/test2_sin_900/';
%out = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/reverb/test1/';
out2 = '/data/liyuy/PROJECTS/DEREVERB3/timit_8k/clean/test6/';
%{
if ~exist(out_path1,'dir')
    mkdir(out_path1);
end


if ~exist(out_path2,'dir')
    mkdir(out_path2)
end

if ~exist(out2,'dir')
    mkdir(out2);
end

if ~exist(out,'dir')
    mkdir(out)
end
%}
if ~exist(out2,'dir')
    mkdir(out2)
end
num_sig = 500;

sigs = 1:num_sig;

Fs = 8e3;

para.SNR        = Inf;
%para.addNoise   = 0;
para.earlyrevb  = 50e-3*Fs;
para.dir_snd    = 1e-3*Fs;
num_peaks       = 5;
min_peak_height = 8e-3;
 
roomNum = 11;

rir_path = sprintf('%s%d.mat',rirPath,roomNum);
fprintf('loading %s...\n',rirPath);
rir1 = load(rir_path);
rir1 = rir1.totalFilter;
fft_length = 1024;
    % Generate reverberant signal
rir = rir1;
len = 45000;
rir_len = 8000;
loc_i = 1;
new_loc = 10;


for sig_i = sigs
        cs_count = sig_i+880;
        
        if sig_i ~= 1 && mod((sig_i - 1),125) == 0
           roomNum = roomNum + 1; 
           %new_loc = 10;
           loc_i = 1;
        end
        
        for new = 1:3
        
        filt = rir{new,loc_i};
        
        filt_speech = filt.hs;
        len_hs = length(filt_speech);
        if len_hs < rir_len
            pad_rir = zeros(1,(rir_len-len_hs));
            filt_speech = horzcat(filt_speech,pad_rir);
        end
        T60 = filt.t60;
        
         cs_path = sprintf('%s%d.wav',data_path,cs_count);
         [cs_data,Fs1] = audioread(cs_path);   
         
        
        
            newLen = length(cs_data);
            if newLen > len
                cs_data = cs_data(1:len);
            elseif newLen < len
                padding = zeros((len-newLen),1);
                cs_data = vertcat(cs_data,padding);
            end
            reverb = conv(cs_data,filt_speech);
            new_len = length(reverb);
            %disp(new_len);
            %save_file    = sprintf('%sreverb_rir%04d_roomNum%d_t60%g_loc%d_%dkHz.wav',out,sig_i,roomNum,T60,loc_i,Fs/1000);
            %audiowrite(save_file,reverb,Fs);
        
        
        
        
       
            
        rir_len     = length(filt_speech);
        [pks,locs]    = findpeaks(filt_speech,'NPEAKS',num_peaks,'SORTSTR','none','MINPEAKHEIGHT',min_peak_height);
        direct_stop = locs(1) + para.dir_snd;
        early_stop  = direct_stop + para.earlyrevb;
                    
        rir_dir   = zeros(size(filt_speech));
        rir_early = zeros(size(filt_speech));
        rir_late  = zeros(size(filt_speech));
                    
        rir_dir(1:direct_stop)              = filt_speech(1:direct_stop);
        rir_early(direct_stop+1:early_stop) = filt_speech(direct_stop+1:early_stop);
        rir_late(early_stop+1:rir_len)      = filt_speech(early_stop+1:rir_len);
        rir_de = rir_dir + rir_early;
        %rir_de = rir_de(1:1024);
        
        disp(early_stop);
        de_fft = fft(rir_de,fft_length);
        de_fft = de_fft(1:513);
        new_reverb = conv(cs_data,rir_de);
        save_file    = sprintf('%sreverb_rir%04d_roomNum%d_t60%g_loc%d_%dkHz.wav',out2,sig_i,roomNum,T60,sig_i,Fs/1000);
        audiowrite(save_file,new_reverb,Fs);
        %save_file    = sprintf('%sreverb_rir%04d_roomNum%d_t60%g_loc%d_%dkHz.wav',out,sig_count,roomNum,T60,sig_count,Fs/1000);
        %audiowrite(save_file,new_reverb,Fs);
        %save(save_file,'de_fft');
        %la_fft = fft(rir_late,fft_length);
        %la_fft = la_fft(1:513);
        
        %end
        end
        loc_i = loc_i + 1;
        new_loc = new_loc + 1;
        %sig_count = sig_count + 1;
        %cs_count = cs_count + 1;
    end

       
        
    

