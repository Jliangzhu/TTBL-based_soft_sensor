clc
clear all 
%*********************************************
%% load data
load SRMG_MI_7001.mat; % Grade1
load SRMG_MI_7501.mat; % Grade2
load SRMG_MI_8001.mat; % Grade3
%% Grade1¡úGrade2 
% grade1_train_y1=SRMG_MI_7001(1:60,13); 
% grade1_train_x1=SRMG_MI_7001(1:60,1:12);
% grade2_test_y=SRMG_MI_7501(31:end,13);
% grade2_test_x=SRMG_MI_7501(31:end,1:12);
% std2=2.4; mu2=16.1;
%*********************************************
%% Grade1¡úGrade3 
% grade1_train_y1=SRMG_MI_7001(1:60,13); 
% grade1_train_x1=SRMG_MI_7001(1:60,1:12);
% grade2_test_y=SRMG_MI_8001(81:end,13);
% grade2_test_x=SRMG_MI_8001(81:end,1:12);
% std2=24.7; mu2=311.3;
%*********************************************
%% Grade2¡úGrade1 
% grade1_train_y1=SRMG_MI_7501(1:60,13); 
% grade1_train_x1=SRMG_MI_7501(1:60,1:12);
% grade2_test_y=SRMG_MI_7001(23:end,13);
% grade2_test_x=SRMG_MI_7001(23:end,1:12);
% std2=13.4; mu2=48.3;
%*********************************************
%% Grade2¡úGrade3 
% grade1_train_y1=SRMG_MI_7501(1:60,13); 
% grade1_train_x1=SRMG_MI_7501(1:60,1:12);
% grade2_test_y=SRMG_MI_8001(81:end,13);
% grade2_test_x=SRMG_MI_8001(81:end,1:12);
% std2=24.7; mu2=311.3;
%*********************************************
%% Grade3¡úGrade1 
% grade1_train_y1=SRMG_MI_8001(1:60,13); 
% grade1_train_x1=SRMG_MI_8001(1:60,1:12);
% grade2_test_y=SRMG_MI_7001(23:end,13);
% grade2_test_x=SRMG_MI_7001(23:end,1:12);
% std2=13.4; mu2=48.3;
%*********************************************
%% Grade3¡úGrade2 
% grade1_train_y1=SRMG_MI_8001(1:60,13); 
% grade1_train_x1=SRMG_MI_8001(1:60,1:12);
% grade2_test_y=SRMG_MI_7501(31:end,13);
% grade2_test_x=SRMG_MI_7501(31:end,1:12);
% std2=2.4; mu2=16.1;
%*********************************************
%% Normalization
MI_s1_train_y1=zscore(grade1_train_y1);
MI_s1_train_x1=zscore(grade1_train_x1);
MI_s2_test_y2=zscore(grade2_test_y);
MI_s2_test_x2=zscore(grade2_test_x);
%% Data combination
train_grade1=[MI_s1_train_y1,MI_s1_train_x1];
test_grade2=[MI_s2_test_y2,MI_s2_test_x2];
Ns=size(MI_s1_train_x1,1); Nt=size(MI_s2_test_x2,1);
Ae=diag([ones(Ns,1);zeros(Nt,1)]); % Empirical matrix Ae
%% TTBL
best_TTBL=TTBL_FLOO_node(train_grade1,test_grade2,Ae,20,22,1,20,22,1,200,400,100);
[TY1]=TTBL(train_grade1,test_grade2);
TY1=TY1*std2+mu2;
RMSE_TTBL=sqrt(mse(TY1-grade2_test_y));
RE_TTBL=sqrt(mse((TY1-grade2_test_y)./grade2_test_y));