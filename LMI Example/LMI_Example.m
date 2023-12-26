close all;
clc;
clear all;

% ENERGY-TO-ENERGY GAIN COMPUTATION USING LMI OPTIMIZATION
% Use of the LMI Toolbox command "mincx' to obtain the solution 
% of the energy-to-energy gain computation
A=[-4 2; 1 -7];
B=[1; 1];
C=[1 0];
D=0;

% Initialize the LMI system to void

setlmis([]);

% Define the variables X (symmetric 2x2 matrix) and gamma (scalar)

P=lmivar(1,[2 1]);
gam=lmivar(1,[1 1]);

% Define the system of LMIs to solve
%
%       ( A'*P + P*A   P*B       C' )
%       ( B'*P      -gam*I       D' )   <  0
%       (    C           D   -gam*I )
%
%                 P   >  0   <==>    -P  <  0
%
%             gamma   >  0   <==> -gamma <  0
%

%
% 1st LMI
%
lmiterm([1 1 1 P],1,A,'s');
lmiterm([1 1 2 P],1,B);
lmiterm([1 1 3 0],C');
lmiterm([1 2 2 gam],-1,1);
lmiterm([1 2 3 0],D');
lmiterm([1 3 3 gam],-1,1);

%
% 2nd LMI
%
lmiterm([2 1 1 P],-1,1);

%
% 3rd LMI
%
lmiterm([3 1 1 gam],-1,1);

%
% Collect the system of LMIs
%
lmisys=getlmis;

%
% Define the cost function c*x to be minimized
c=mat2dec(lmisys,zeros(2,2),1);
%
% Perform the LMI minimization
%
[cost x_opt]=mincx(lmisys,c);
%
% Compute the optimal values of the parameters
%
P_opt = dec2mat(lmisys,x_opt,P);
gam_opt = dec2mat(lmisys,x_opt,gam);

% Energy-to-energy gain
GAMMA_ee=gam_opt
%
% Check the results - evaluate the system of LMIs
LMI_eval=evallmi(lmisys,x_opt);
[LMI1_lhs,LMI1_rhs]=showlmi(LMI_eval,1);
[LMI2_lhs,LMI2_rhs]=showlmi(LMI_eval,2);
[LMI3_lhs,LMI3_rhs]=showlmi(LMI_eval,3);

%
% Check that LMI1_lhs < 0, LMI2_lhs < 0, LMI3_lhs < 0 %
%
check1=eig(LMI1_lhs)
check2=eig(LMI2_lhs)
check3=eig(LMI3_lhs)

%
% Check the energy-to-energy gain answer using 'norminf'
%
sys=ltisys(A,B,C,D);
Hinf_norm=norminf(sys)
