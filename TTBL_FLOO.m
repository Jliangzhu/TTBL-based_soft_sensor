function [H,WFSparse,wh,BiasofEnhanceNeurons] = TTBL_FLOO(X,N1,N2,N3)
y=zeros(size(X,1),N2*N1);
ind=ones(size(X,1),1);
BiasofEnhanceNeurons=rand(1,N3);
BiasMatrix=BiasofEnhanceNeurons(ind,:);
for i=1:N2
    we=2*rand(size(X,2),N1)-1; % we is the random weights including bias
    A1=X * we; % A1=mapminmax(A1); % A1 is the output of feature nodes in each window
    beta1=sparse_bls(A1,X,1e-3,50)'; % seek the sparse weights
    T1=X * beta1;
    WFSparse{i}=beta1;
    y(:,N1*(i-1)+1:N1*i)=T1;                                
end
y=tansig(y);
% %%%%%%%%%enhancement nodes%%%%%%%%% 
% if N1*N2>=N3                                                              
%      wh=orth(2*rand(N2*N1,N3)-1);
% else
%     wh=orth(2*rand(N2*N1,N3)'-1)'; 
% end
wh=2*rand(N2*N1,N3)-1;
T2=y*wh+BiasMatrix;
T2=tansig(T2);%enhancement nodes 
T3=[y T2];%feature nodes and enhancement nodes 
H=T3;














