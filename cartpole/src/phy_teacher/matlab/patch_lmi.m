function [F_hat, tmin] = patch_lmi(Ak, Bk, Z)
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

% C = [1/10,     0,        0,        0;
        % 0,   1/10000,       0,        0;
        % 0,     0,     1/10,        0;
        % 0,     0,        0,     1/10000];

% C = [1/1.5,     0,        0,        0;
%          0,     0,      1/1,        0];

C = [1/2,     0,        0,        0;
       0,     0,      1/2,        0];

D = 1/50;

Z = diag(Z);

n = 4;
alpha = 0.999;
phi = 0.0001;

setlmis([]) 
Q = lmivar(1,[4 1]); 
R = lmivar(2,[1 4]);

T = lmivar(1,[1 1]); 

lmiterm([1 1 1 Q], C, C');
lmiterm([1 1 1 0], -eye(2));

lmiterm([2 1 1 T], D, D');
lmiterm([2 1 1 0], -eye(1));

lmiterm([-3 1 1 Q], 1, 1);
lmiterm([-3 1 1 0], -n*Z*Z);

lmiterm([-4 1 1 Q], 1, alpha);
lmiterm([-4 2 1 Q], Ak, 1);
lmiterm([-4 2 1 R], Bk, 1);
lmiterm([-4 2 2 Q], 1, 1 / (1+phi));

lmiterm([-5 1 1 Q], 1, 1);
lmiterm([-5 2 1 R], 1, 1);
lmiterm([-5 2 2 T], 1, 1);

mylmi = getlmis;

[tmin, psol] = feasp(mylmi);
% assert(tmin < 0)

Q = dec2mat(mylmi, psol, Q);
R = dec2mat(mylmi, psol, R);
F_hat = R*inv(Q);

M = Ak + Bk*F_hat;
% assert(all(eig(M)<1))

end