close all; clear all; clc;

load HkforTiming.mat

rhs = randn(size(Hk,2),1) + 1i*randn(size(Hk,2),1);

%%

tic; 
t1 = Hk\rhs;
toc;

%%

[LL,UU,Pp,Qp,Rr] = lu(Hk);       
tic;
t2 = Qp*(UU\(LL\(Pp*(Rr\(rhs)))));             
toc;
norm(t1-t2)/norm(t1)
%%
[IL,JL,VL] = find(LL);
[IU,JU,VU] = find(UU);
[IP,JP,VP] = find(Pp);
[IQ,JQ,VQ] = find(Qp);
[IR,JR,VR] = find(Rr);
[n1,n2] = size(Hk);
tic; 
Lt = sparse(IL,JL,VL,n1,n2);
Ut = sparse(IU,JU,VU,n1,n2);
Pt = sparse(IP,JP,VP,n1,n2);
Qt = sparse(IQ,JQ,VQ,n1,n2);
Rt = sparse(IR,JR,VR,n1,n2);
t2 = Qt*(Ut\(Lt\(Pt*(Rt\(rhs)))));             
toc;
norm(t1-t2)/norm(t1)

