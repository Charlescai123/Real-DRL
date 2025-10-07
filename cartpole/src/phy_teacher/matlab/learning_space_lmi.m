clear;
clc;
D = [1/2, 0, 0, 0;
     0, 0, 1/2, 0
];
cvx_begin sdp
variable Q(4,4) symmetric;
minimize -log_det(Q)
D * Q * D' - eye(2) < 0
Q > 0
cvx_end
P = round(inv(Q),4);
writematrix(P, "P1.txt");