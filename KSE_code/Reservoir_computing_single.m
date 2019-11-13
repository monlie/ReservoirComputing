clc
clear all

tic;
load('KS_data_right.mat');

udata=udata';

siz=size(udata);
final_rmse=[];
count=1;


%% define trainning data and validation data
disp('Preparing training and validation data process......')
train_X=udata(2001:72010,:)'; train_Y=udata(2002:72011,:)';
%Validate_X=U_all(8001:8301,:)'; 
Validate_Y=udata(72012:73211,:)';

final_Diff_Y=[];
count_spec=0;
for spec=-3:0.3:0.3
    count_spec=count_spec+1;
%% define res_net
disp('creating reservoir_network process......')
n=4992; k=3;%16384

res_net=[];
load('res_net_example_1.mat');
res_net=res_net*(10^spec);
%res_net = gpuArray(res_net);% if you want use gpu using this line
toc;
%% define W_in
disp('Creating input network process......')
W_in=[];
strs = 2;
index=randperm(n); index=reshape(index,n/siz(1,2),siz(1,2));
for i=1:siz(1,2)
    W_in(index(:,i),i)=strs*rand(n/siz(1,2),1)-(strs/2);
end
%W_in = gpuArray(W_in);
toc;

%% tranning resnet model
disp('Begin training process......')
beta=1*10^(-4);
r_all=[];
P=[];
r_all(:,1)=zeros(n,1);%2*rand(n,1)-1;%
r_input=[];
x_input=[];
for i=1:size(train_X,2)
%    disp(i);
    r_input(:,i)=res_net*r_all(:,i);
    x_input(:,i)=W_in*train_X(:,i);
    r_all(:,i+1)=tanh(r_input(:,i)+x_input(:,i));
end
r=r_all(:,12:end);
r_end=r_all(:,end); 
r(2:2:end,:)=r(2:2:end,:).^2;
%r=[r;train_X(:,11:end)];
% r=r+r.^2+r.^3;
P = train_Y(:,11:end)*r'*(r*r'+beta*eye(n))^(-1);%+size(udata,2)

diff_train=train_Y(:,11:end)-P*r;

toc
%% resnet model predicting
disp('Begin predicting process......')
predict_Y=[];r_1=r_end;%rand(n,1);% %r(:,1);%

X_1=train_Y(:,end);%train_X(:,12);%
r_pred_all=[];
%u_0=eye(n);
h=[];
%DM=eye(n);
for j=1:size(Validate_Y,2)
    disp(j)
    r_pred=[];
    value = res_net*r_1+W_in*X_1;
%     temp_v=2.*r_1;
%     temp_v(1:2:end-1)=1;
%     DM=DM*((res_net+W_in*P*diag(temp_v))*diag((sech(value)).^2));
    r_pred=tanh(value);
    r_1=r_pred;
    r_pred_all(:,j)=r_1;
    r_pred(2:2:end,1)=r_pred(2:2:end,1).^2;
%    r_pred=[r_pred;X_1];
%     r_pred=r_pred+r_pred.^2+r_pred.^3;
    predict_Y(:,j)=P*r_pred;
    X_1=predict_Y(:,j);%Validate_Y(:,j);%
    
%     h(:,j)=sort(log(vecnorm(gather(DM*u_0)))/j)';
    
end
Diff_Y=Validate_Y-predict_Y;
final_Diff_Y(:,:,count_spec)=Diff_Y;
end

siz_1=size(final_Diff_Y);
final_Diff_Y_rmse = reshape(sqrt(mean(final_Diff_Y.^2)),siz_1(2),siz_1(3));
save('res_net_example_1_random_k_3_uew_rho_10_to_-3_0.3_0.3.mat','-v7.3')

%% Drawing result

figure;
subplot(3,1,1);
imagesc(((1:size(Validate_Y,2))*0.25*0.05),(22/64)*(-64/2:64/2-1),Validate_Y);colormap jet;%(2,-2):*0.154,(2,-6):*0.3740
subplot(3,1,2);
imagesc(((1:size(Validate_Y,2))*0.25*0.05),(22/64)*(-64/2:64/2-1),predict_Y);colormap jet;% colorbar
subplot(3,1,3);
imagesc(((1:size(Validate_Y,2))*0.25*0.05),(22/64)*(-64/2:64/2-1),Diff_Y);colormap jet;

figure;
imagesc(final_Diff_Y_rmse);
toc


%% code for generating resvoir network
% n=4992;k=3;% n is the number of neuron, k is the average degree
% index1=repmat(1:n,1,k)';
% index2=randperm(n*k)';
% index2(:,2)=repmat(1:n,1,k)';
% index2=sortrows(index2,1);
% index1(:,2)=index2(:,2);
% res_net=sparse(index1(:,1),index1(:,2),rand(size(index1,1),1)',n,n);
% eig_D=eigs(res_net);
% res_net=(spec/(abs(eig_D(1,1)))).*res_net;% spec is the spectral radius
% res_net=full(res_net);