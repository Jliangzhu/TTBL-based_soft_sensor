function [best] = TTBL_FLOO_node(Xtrain_s,Xtest_t,Ae,Numberofnode1,Numberofnode2,DNumberofnode1,Numberofnode3,Numberofnode4,DNumberofnode2,Numberofnode5,Numberofnode6,DNumberofnode3)

%%------------------1.Loading datasets from different domains-------------------------
YX1=Xtrain_s(:,1);  % The label in source domain
YX2=zeros(size(Xtest_t,1),1);  % The 0 matrix with corresponding size
Xs=Xtrain_s(:,2:end);  % The input data in source domain
Xt=Xtest_t(:,2:end);  % The input data in target domain
Ns=size(Xs,1); Nt=size(Xt,1); X=[Xs;Xt]; 

%%------------------2.The ranges of parameter selection-------------------------------
best=50; Gamma=[0.05,0.1,0.15]; k1=[5,7,9];
ND1=(Numberofnode2 - Numberofnode1)/DNumberofnode1 + 1;
ND2=(Numberofnode4 - Numberofnode3)/DNumberofnode2 + 1;
ND3=(Numberofnode6 - Numberofnode5)/DNumberofnode3 + 1;
Numberofnode1=linspace(Numberofnode1,Numberofnode2,ND1); % feature nodes in each group
Numberofnode2=linspace(Numberofnode3,Numberofnode4,ND2); % feature node groups
Numberofnode3=linspace(Numberofnode5,Numberofnode6,ND3); % enhancement nodes

disp('Start the parameters selection by FLOO-CV...');
%%------------------------3.建立FlOO_Lamda模型-----------------------------
disp('Select the parameter k for TTBL...');
for m=1:3
    for t1=1:ND1
        disp('Select the parameter ND1 for TTBL...');
        for t2=1:ND2
            disp('Select the parameter ND2 for TTBL...');
            for t3=1:ND3
                disp('Select the parameter ND3 for TTBL...');
                [H,InputWeight,wh,BiasofEnhanceNeurons]=TTBL_FLOO(X,Numberofnode1(t1),Numberofnode2(t2),Numberofnode3(t3));
                manifold.k=k1(1,m);
                manifold.Metric='Cosine';
                manifold.NeighborMode='KNN';
                manifold.WeightMode='Cosine';
                W=construct_graph(X,manifold);
                D=diag(sparse(sqrt(1./sum(W))));
                L=speye(Ns+Nt)-D*W*D;
                for i=1:3
                    disp('Select the parameter Gamma for TTBL...'); 
                    H0=(Ae*H*H'+ones(size(H*H'))+Gamma(i)*L*H*H'); 
                    YX=[YX1;YX2];
                    % Hs=H(1:Ns,:); YX=YX1; 
                    % H0=(Hs'*Hs+ones(size(Hs'*Hs))+Gamma(i)*H'*L*H);
                    for j=1:Ns
                        yi=YX(j,:);
                        Hi=H(j,:);
                        HJi=Hi*H'*pinv(H0)*Ae; 
                        P_yi(j,:)=(HJi*YX - HJi(:,j)*yi)/(1-HJi(:,j)); 
%                       HJi=Hi*pinv(H0)*Hs';
%                       P_yi(j,:)=(HJi*YX - HJi(:,j)*yi)/(1-HJi(:,j));
                        % Calculate the i-th cross validation error
                        ri(j,:)=abs(yi-P_yi(j,:));
                    end
                    % Calculate the mean square error
                    A=(ri).^2;
                    b=sum(A);
                    RMSE=sqrt(b/Ns);
                    Numberofnode11=Numberofnode1(t1);
                    Numberofnode21=Numberofnode2(t2);
                    Numberofnode31=Numberofnode3(t3);
                    Gamma1=Gamma(i);
                    K=k1(1,m);
                    if best>RMSE
                        disp('save parameters...');
                        best=RMSE;
                        save('optimal_TTBL.mat','BiasofEnhanceNeurons','InputWeight','wh','Numberofnode11','Numberofnode21','Numberofnode31','Gamma1','L','Ae','K');
                    end
                end
            end
        end
    end
end






