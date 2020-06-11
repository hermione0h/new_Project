clear
num_rir = 125;
a=load('test6_0.9_denoising.mat');
scores = a.scores_denoise_fcIRM;
pesq_scores = [];
stoi_scores = [];
for i =1:num_rir
    score = scores{i,1};
    pesq_scores(i) = score.pesq_derev;
    stoi_scores(i) = score.stoi_derev;
end
ave_pesq = sum(pesq_scores)/num_rir;
std_pesq = std(pesq_scores);
ave_stoi = sum(stoi_scores)/num_rir;
std_stoi = std(stoi_scores);
disp(ave_pesq);
disp(std_pesq);
disp(ave_stoi);
disp(std_stoi);