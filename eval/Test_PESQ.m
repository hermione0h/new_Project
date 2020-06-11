clear all; close all; clc;

    % name of executable file for PESQ calculation
    binary = 'pesq2.exe';

    % specify path to folder with reference and degraded audio files in it
    pathmix = 'C:\Users\Zhang Zhuohuang\Box Sync\Research\Algorithm Compare\Codes\LogMMSE\Predicted';
    pathclean = 'C:\Users\Zhang Zhuohuang\Box Sync\Research\Algorithm Compare\Codes\LogMMSE\Upsampled_Clean';

    % specify reference and degraded audio files
    reference = 'sp01.wav';
    degraded = 'Predicted_sp01_babble_sn10.wav';

    
    % compute NB-PESQ and WB-PESQ scores for wav-files 0dB SNR
    nb = pesq2_mtlb2( reference, degraded, 16000, 'nb', binary, '.\', '.\clean' );
    wb = pesq2_mtlb2( reference, degraded, 16000, 'wb', binary, '.\', '.\clean' );
    
    
    % display results to screen
    fprintf('====================\n'); 
%     disp('Compute NB-PESQ scores for wav-files (0dB SNR):');
    fprintf( 'NB PESQ MOS = %5.3f\n', nb(1) );
    fprintf( 'NB MOS LQO  = %5.3f\n', nb(2) );
    
    
    fprintf('====================\n'); 
%     disp('Compute WB-PESQ score for wav-files (0dB SNR):');
    fprintf( 'WB MOS LQO  = %5.3f\n', wb );
