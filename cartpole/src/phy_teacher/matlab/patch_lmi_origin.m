function [F_hat, tmin] = patch_lmi_origin(chi, Ak, Bk)
%%%%%%%%%%%%%%%%%%%%%%  DOC HELP  %%%%%%%%%%%%%%%%%%%%%%
%% Inputs
%
%      chi:  Hyperparameter for patch
%      Ak :  A(s) in discrete form        -- 4x4
%      Bk :  B(s) in discrete form        -- 4x1
%
%% Outputs
%   F_hat :  Feedback control gain        -- 1x4
%    tmin :  Feasibility of LMI solution  -- 1x4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P = [54.1134178606985,26.2600592637275,61.7975412804215,12.9959418258126;
      26.2600592637275,14.3613985149923,34.6710819094179,7.27321583818861;
      61.7975412804215,34.6710819094179,88.7394386456256,18.0856894519164;
      12.9959418258126,7.27321583818861,18.0856894519164,3.83961074325448;];

D = [1/1,     0,        0,        0;
       0,     0,      1/0.8,      0];
C = 1/50;

n = 4;
alpha = 0.999;
kappa = 0.01;
gamma1 = 1;
gamma2 = 0.1;


setlmis([]) 
Q = lmivar(1,[4 1]); 
R = lmivar(2,[1 4]);
mu = lmivar(2,[1 1]);
T = lmivar(1,[1 1]); 

lmiterm([-1 1 1 Q],1,(alpha - kappa*(1+(1/gamma2)))*eye(4));
lmiterm([-1 2 1 Q],Ak,1);
lmiterm([-1 2 1 R],Bk,1);
lmiterm([-1 2 2 Q],1,1/(1+gamma2));

lmiterm([2 1 1 mu], 1, -1);

lmiterm([3 1 1 mu], 1,inv(P));
lmiterm([3 1 1 Q], 1,-1);

lmiterm([4 1 1 0], (1-(2*chi)+(chi/gamma1))*1);
lmiterm([4 1 1 mu], 1, (chi*gamma1)-1);

lmiterm([5 1 1 Q], D, D');
lmiterm([5 1 1 0], -eye(2));

lmiterm([-6 1 1 Q],1,1);
lmiterm([-6 2 1 R],1,1);
lmiterm([-6 2 2 T],1,1);

lmiterm([7 1 1 T], C, C');
lmiterm([7 1 1 0], -eye(1));

mylmi = getlmis;

[tmin, psol] = feasp(mylmi);
% assert(tmin < 0)

Q = dec2mat(mylmi, psol, Q);
R = dec2mat(mylmi, psol, R);
F_hat = R*inv(Q);

M = Ak + Bk*F_hat;
assert(all(eig(M)<1))

end