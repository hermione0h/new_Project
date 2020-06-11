addpath('/home/liyuy/PESQ/');

path1 = '/data/liyuy/output_fm_25_3/';
path2 = '/data/liyuy/test2/';
list = dir(strcat(path1, '*.wav'));
list2 = dir(strcat(path2, '*.wav'));
listLength = length(list);
desiredFrequency = 16000;
x = [];
%path3 = '/data/liyuy/newnew/';
for i = 1:listLength
    yNew = [];
    targetPath = strcat(path2, list2(i).name);
    [y, fs] = audioread(targetPath);
    yNew = resample(y,desiredFrequency,fs);
    %filename = sprintf('%s%s.%s',path3,num2str(i),'wav');
    %audiowrite(filename,yNew,desiredFrequency);
    targetPath2 = strcat(path1,list(i).name);
    [y1,fs1] = audioread(targetPath2);
    score = pesqscore(y1,yNew); 
    x = [x score];
end
avescore = sum(x)/listLength;
stdnum = std(x);
disp(stdnum);
disp(avescore);

