function [score] = pesqscore(vector1, vector2,Fs)


if length(vector1) <= length(vector2)
    vector2 = vector2(1:length(vector1));
else
    vector1 = vector1(1:length(vector2));
end
score = pesq_dat(vector1,vector2,Fs);