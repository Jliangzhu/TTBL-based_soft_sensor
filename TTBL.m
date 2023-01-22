function [TY] = TTBL(TrainingData, TestingData)

%%%%%%%%%%% Load the model parameters
load optimal_TTBL.mat;
Gamma=Gamma1;
Ct=Gamma;
N1 = Numberofnode11;
N2 = Numberofnode21;
N3 = Numberofnode31;
%%%%%%%%%%% Load dataset from source domain
train_data=TrainingData;
S=train_data(:,1);
P=train_data(:,2:size(train_data,2));
clear train_data;  % Release raw training data array
%%%%%%%%% Load dataset from target domain
Pt=TestingData(:,2:end);
Ns=size(S,1); Nt=size(Pt,1);
X=[P;Pt];
inds=ones(size(X,1),1);
BiasMatrixS=BiasofEnhanceNeurons(inds,:);
for i=1:N2
    InputWeight1=InputWeight{i};
    y=X*InputWeight1;
    tempH(:,N1*(i-1)+1:N1*i)=y;
end
Z=tansig(tempH);
H=tansig(Z*wh+BiasMatrixS);
H_X=[Z,H]; 
H_Xs=H_X(1:Ns,:); H_Xt=H_X((Ns+1):end,:);
Btail=(speye(size(H_X',1))+Ct*(H_X'*L*H_X)+(H_Xs'*H_Xs))\(H_Xs'*S);
TY=H_Xt*Btail;
% Y=[T;zeros(size(Pt,1),1)];
% Btail=(speye(Ns+Nt) +Ct*L*(H_X*H_X')+ Ae*(H_X*H_X')) \ (Ae*Y);
% B=H_X'*Btail; Yhat=H_X*B;
% TY=Yhat((Ns+1):end,:);