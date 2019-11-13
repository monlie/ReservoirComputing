% code for generating resvoir network
n=5;k=3;% n is the number of neuron, k is the average degree
index1=repmat(1:n,1,k)';
index2=randperm(n*k)';
index2(:,2)=repmat(1:n,1,k)';
index2=sortrows(index2,1);
index1(:,2)=index2(:,2);
res_net=sparse(index1(:,1),index1(:,2),rand(size(index1,1),1)',n,n);
eig_D=eigs(res_net);
res_net=(spec/(abs(eig_D(1,1)))).*res_net;% spec is the spectral radius
res_net=full(res_net);