function [F_kp, F_kd, tmin] = patch_lmi(tracking_error)
    
%%%%%%%%%%%%%%%%%%%%%%  DOC HELP  %%%%%%%%%%%%%%%%%%%%%%
%% Inputs
%
%   tracking_err: System tracking error
%
%% Outputs
%    F_kp :  Feedback control gain        -- 6x6
%    F_kd :  Feedback control gain        -- 6x6
%    tmin :  Feasibility of LMI solution  -- 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

roll = tracking_error(2);
pitch = tracking_error(3);
yaw = tracking_error(4);
%%Rotation Matrix%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rx = [1,    0,         0;
      0,    cos(roll), -sin(roll);
      0,    sin(roll), cos(roll)];

Ry =  [cos(pitch),  0,   sin(pitch);
       0,           1,   0;
       -sin(pitch), 0,  cos(pitch)];

Rz = [cos(yaw),  -sin(yaw),  0;
      sin(yaw),  cos(yaw),   0;
      0,         0,          1];

Rzyx = Rz*Ry*Rx;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Sampling period%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = 1 / 30;      %work in 25 to 30
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%System matrices (continuous-time)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
aA = zeros(10, 10);
aA(1,7) = 1;
aA(2:4, 8:10) = Rzyx;
aB = zeros(10, 6);
aB(5:10, :) = eye(6);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%System matrices (discrete-time)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = aB * T;
A = eye(10) + T * aA;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%safety conditions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cc = 0.6;
b1 = 1 / 0.8;         %yaw
b2 = 1 / (1.0 * cc);  %height
b3 = 1 / 1.5;         %velocity
b4 = 1 / 1;

D = [b2, 0, 0,  0,  0,  0,  0,  0,  0,  0;
      0, 0, b4, 0,  0,  0,  0,  0,  0,  0;
      0, 0, 0,  0,  b3, 0,  0,  0,  0,  0;
      0, 0, 0,  b1, 0,  0,  0,  0,  0,  0];

c1 = 1 / 30;
c2 = 1 / 60;

C = [c1,  0,  0,  0,  0,  0;
      0,  c1, 0,  0,  0,  0;
      0,  0, c1,  0,  0,  0;
      0,  0,  0,  c2, 0,  0;
      0,  0,  0,  0,  c2, 0;
      0,  0,  0,  0,  0,  c2];

Z = diag(tracking_error);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%LMI parameter%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.9;
n = 10;
phi = 0.2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%LMI formulas%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
setlmis([])
Q = lmivar(1,[10 1]);
T = lmivar(1,[6 1]);
R = lmivar(2,[6 10]);

% 1st LMI
lmiterm([-1 1 1 Q],1,alpha);
lmiterm([-1 2 1 Q],A,1);
lmiterm([-1 2 1 R],B,1);
lmiterm([-1 2 2 Q],1,1/(1 + phi));

% 2nd LMI
lmiterm([-2 1 1 Q],1,1);
lmiterm([-2 2 1 R],1,1);
lmiterm([-2 2 2 T],1,1);

% 3rd LMI
lmiterm([-3 1 1 Q],1,1);
lmiterm([-3 1 1 0],-10*Z*Z);

% 4th LMI
lmiterm([-4 1 1 Q],-D,D');
lmiterm([-4 1 1 0],eye(4));

% 5th LMI
lmiterm([-5 1 1 T],-C,C');
lmiterm([-5 1 1 0],eye(6));

mylmi = getlmis;

[tmin, psol] = feasp(mylmi);
Q = dec2mat(mylmi, psol, Q);
R = dec2mat(mylmi, psol, R);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P = inv(Q);
aF = aB * R * P;
Fb2 = aF(7:10, 1:4);

F_kp = -[zeros(2, 6); zeros(4, 2), Fb2]
F_kd = -aF(5:10, 5:10)

% M = aA + aF;
% real(eig(M))
% assert(all(real(eig(M))<0))

end