 function [A,E,H,P,Q,W,Z,obj] = LRMPL(X,Y,l,lambda_1,lambda_2,lambda_3)
%%  the objective function: min_{A,E,H,P,Q,W,Z} ||L-AX||_F^2 + lambda_1 * ||Z||_* + lambda_2 * ||E||_{2,1} + lambda_3 * ||A||_F^2
Max_iter = 10;
mu = 0.1;
rho = 1.01;
max_mu = 10^6;
[m,n] = size(X);
c = length(unique(l));
L = zeros(c,n);
loc = 0;

for i = 1:c
    num = sum(l(:) == i);
    L(i,loc + 1 : loc + num) = 1;
    loc = loc + num;
end  %Generating tag matrix one-hot code

%%------------------------------initilzation-------------------------------
options = [];
options.ReducedDim = c;
[P1,~] = PCA1(X',options);

W = eye(c);
Q = ones(m,c);
A = W'*Q';
E = ones(m,n);
H = ones(n,n);
Z = ones(n,n);

Y1 = zeros(m,n);
Y2 = zeros(n,n);
Y3 = zeros(c,m);
%%-------------------------end of initilazation----------------------------
for iter = 1:Max_iter
%      iter
    % P
    if (iter == 1)
        P = P1;
    else
        K3 = X-E+Y1/mu;
        [U1,~,S1] = svd(K3*Z'*Y'*Q,'econ');   
        P = U1*S1';
        clear K3;
    end
    % A
    K1 = W'*Q' + Y3/mu;
    A = (2*L*X'+mu*K1)/(2*X*X'+(2 * lambda_3 + mu)*eye(m)+0.01*eye(m));  
    % E
    [~,j] = size(E);
    K2 = X - P*Q'*Y*Z + Y1/mu;
    for i = 1:j
        K_n2 = norm(K2(:,i),2);
        if K_n2 > lambda_2 / mu
            E(:,i) = ( (K_n2 - lambda_2 / mu) / K_n2 ) * K2(:,i);
        else
            E(:,i) = 0;
        end
    end
    % H
    eps1 = lambda_1/mu;
    temp_A = Z+Y2/mu;
    [uu,ss,vv] = svd(temp_A,'econ');
    ss = diag(ss);
    SVP = length(find(ss>eps1));
    if SVP>=1
        ss = ss(1:SVP)-eps1;
    else
        SVP = 1;
        ss = 0;
    end
    H = uu(:,1:SVP)*diag(ss)*vv(:,1:SVP)';
    clear temp_A; clear eps1;
    % Q 
    K4 = X-E+Y1/mu;
    K5 = Y3/mu - A;
    AA = Y*Z*Z'*Y'; 
    BB = W*W';
    CC = K5'*W' - Y*Z*K4'*P;
    Q = lyap(AA,BB,CC); 
    clear K4; clear K5; clear AA; clear BB; clear CC;
    % W 
    K6 = Y3/mu - A;
    W = Q \ (-K6');
    clear K6;
    % Z 
    K7 = X-E+Y1/mu;
    K8 = Y2/mu - H;
    Z = ((Y'*Q*Q'*Y)+eye(n)) \ (Y'*Q*P'*K7-K8+0.01*eye(n));
    clear K7; clear K8;
    
    % Y1;Y2;mu
    Y1 = Y1+mu*(X-P*Q'*Y*Z-E);
    Y2 = Y2+mu*(Z-H);
    Y3 = Y3+mu*(W'*Q'-A);
    mu = min(rho*mu,max_mu);
    leq1 = norm(X-P*Q'*Y*Z-E,Inf);
    leq2 = norm(Z-H,Inf);
    leq3 = norm(W'*Q'-A,Inf);
    %ee = sum(abs(E),2);
    obj(iter) = norm(L-A*X,'fro')^2+lambda_1*sum(svd(H))+lambda_2*sum(sqrt(sum(E.*E,2)))+lambda_3*norm(W,'fro')^2;

    if iter > 2     
        if leq1 < 10^-5 %&& leq2 < 10^-5 && leq3 < 10^-5 && abs(obj(iter)-obj(iter-1)) < 10^-3
            break
        end
    end   
end