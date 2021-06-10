function [R,R_label] = pre(X,Y,X_label,Y_label)

[L_number,~] = size(unique(X_label));

%[m,n_1] = size(X);
%[~,n_2] = size(Y);
%n = n_1 + n_2;
%R = zeros(m,n);
R = [];
R_label = [];
for i = 1:L_number-1
   j = 1;
   while(X_label(j) == i)
        j = j + 1;
   end
   R = [R;X(1:j-1, :)];
   R_label = [R_label;X_label(1:j-1)];
   X_label(1:j-1) = [];
   X(1:j-1, :) = [];
   k = 1;
   while(Y_label(k) == i)
        k = k + 1;
   end
   R = [R;Y(1:k-1, :)];
   R_label = [R_label;Y_label(1:k-1)];
   Y(1:k-1, :) = [];
   Y_label(1:k-1) = []; 
end

[n_1,~] = size(X_label);
[n_2,~] = size(Y_label);

R = [R;X(1:n_1, :);Y(1:n_2, :)];

R_label = [R_label;X_label;Y_label];
% 拼接矩阵,每一行为一个实例对象，每一列为一个特征。
