clear;
clc;

fd = fopen('result.txt','w');  % Save the experimental results
lambda_1 = [1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1];
lambda_2 = [1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1];
lambda_3 = [1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1];
 
 a = {'amazon_SURF_L10','Caltech10_SURF_L10','dslr_SURF_L10','webcam_SURF_L10'};
 b = strcat('./data/Office & Caltech256/',a);
 
 for nu_1 = 1:9
    for nu_2 = 1:9
        for nu_3 = 1:9
        l1 = lambda_1(nu_1);
        l2 = lambda_2(nu_2);
        l3 = lambda_3(nu_3);
        fprintf(fd,'lambda_1 = %2.2d, lambda_2 = %2.2d, lambda_3 = %2.2d \n', l1, l2 , l3);
 
        for i = 1:4
            for j = 1:3
                acc = [];
                tic;
                for v = 1:20
                    clearvars -except fd a b i j v acc lambda_1 lambda_2 lambda_3 l1 l2 l3 nu_1 nu_2 nu_3;
                    
                    if(rem(i+j,4) == 0)
                        load(b{4});
                    else
                        load(b{rem(i+j,4)});
                    end
                    
                    c = length(unique(labels));
                    f = [];
                    g = [];
                    
                    for k = 1:c
                        num = find(labels(:) == k);
                        f = [f;num(randperm(size(num,1),8))];
                    end
                    
                    features = fts(f,:);
                    final_labels = labels(f);
                    
                    fts(f,:) = [];
                    labels(f) = [];
                    
                    test_fts = fts./repmat(sqrt(sum(fts.^2,2)),[1 size(fts,2)]);
                    test_labels = labels;
                    
                    load(b{i});
                    
                    for k = 1:c
                        num = find(labels(:) == k);
                        if( i == 1)
                            g = [g;num(randperm(size(num,1),20))];
                        else
                            g = [g;num(randperm(size(num,1),8))];
                        end
                    end
                    
                    fts = fts(g,:);
                    labels = labels(g);
                    
                    
                    [X,l] = pre(fts, features, labels,  final_labels);
                    [Y,~] = pre(features, fts,  final_labels, labels);
                    
                    X = X./repmat(sqrt(sum(X.^2,2)),[1 size(X,2)]);
                    Y = Y./repmat(sqrt(sum(Y.^2,2)),[1 size(Y,2)]);
                    
                    [A,~,~,~,~,~,~,~] = LRMPL(X',Y',l,l1,l2,l3);
                    
                    [~,predict_label] = max(test_fts * A',[],2);
                    accuracy = length(find(predict_label == test_labels)) / length(test_labels) * 100;
                    acc(v) = accuracy;
                    
                end
                time = toc;
                    if(rem(i+j,4) == 0)
                        fprintf(fd,'\t  %s->%s: accuracy = %2.2f, std = %2.2f, time = %2.2f \n',a{i},a{4},mean(acc),std(acc,1),time);
                    else
                        fprintf(fd,'\t  %s->%s: accuracy = %2.2f, std = %2.2f, time = %2.2f \n',a{i},a{rem(i+j,4)},mean(acc),std(acc,1),time);
                    end
          
            end
        end
        end
    end
end

fclose(fd);