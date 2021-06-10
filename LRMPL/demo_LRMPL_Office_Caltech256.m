clear;
clc;

%----------------------------Super parameter setting-----------------------
lambda_1 = 1e-6;
lambda_2 = 1e-5;
lambda_3 = 1;  %you can change the values of these super parameters to experiment
%--------------------------------------------------------------------------

%----------------------------load data------------------------------------
train = load('data\Office & Caltech256\Caltech10_SURF_L10');
test = load('data\Office & Caltech256\webcam_SURF_L10');   % you can add your data set to the './data' file for experiments
%----------------------------End of import data--------------------------

%--------------------------perform experiments---------------------------
for v = 1:20
                    train_data = train;
                    test_data = test;
                    fprintf(' the %.0fth time experiment\n\n',v );
                    c = length(unique(test_data.labels));
                    f = [];
                    g = [];
                    
                    for k = 1:c
                        num = find(test_data.labels(:) == k);
                        f = [f;num(randperm(size(num,1),8))];
                    end      %Randomly extract i data samples from each class of the target domain, (here i = 8)
                    
                    features = test_data.fts(f,:);
                    final_labels = test_data.labels(f);  % Selected the data sample from the target domain to construct the new domain
                    
                    test_data.fts(f,:) = [];
                    test_data.labels(f) = [];  %remove the data samples used to construct the new domain
                    
                    test_fts = test_data.fts./repmat(sqrt(sum(test_data.fts.^2,2)),[1 size(test_data.fts,2)]);
                    test_labels = test_data.labels;  %
                    
                    
                    
                    for k = 1:c
                        num = find(train_data.labels(:) == k);
%                         if( i == 1)
%                             g = [g;num(randperm(size(num,1),20))];%where Amazon is the source domain, 
%                         else
                            g = [g;num(randperm(size(num,1),8))];%where Amazon is not the source domain.
%                         end
                    end   %Randomly extract i data samples from each class of the target domain, (here i = 8),
                    
                    
                    fts = train_data.fts(g,:);
                    labels = train_data.labels(g); % Selected the data sample from the source domain to construct the new domain
                    
                    [X,l] = pre(fts, features, labels,  final_labels);
                    [Y,~] = pre(features, fts,  final_labels, labels);  %Construct two new data domain X and Y.
                    
                    X = X./repmat(sqrt(sum(X.^2,2)),[1 size(X,2)]);
                    Y = Y./repmat(sqrt(sum(Y.^2,2)),[1 size(Y,2)]);   
                    
                    [A,~,~,~,~,~,~,~] = LRMPL(X',Y',l,lambda_1,lambda_2,lambda_3);   %Obtain the classifier matrix A.
                    
                    [~,predict_label] = max(test_fts * A',[],2);
                    accuracy = length(find(predict_label == test_labels)) / length(test_labels) * 100;
                    acc(v) = accuracy;
end
                    final_accuracy = mean(acc);
%-------------final_accuracy is the average of twenty experiments-----------           
             fprintf(' %.2f accuracy \n\n',final_accuracy);
 
