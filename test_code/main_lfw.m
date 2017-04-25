%% main
clear;
clc
savepath = 'affine_p';
addpath('./utils');
mkdir([savepath,'/pubdata']);
mkdir([savepath,'/models']);
mkdir([savepath,'/results']);

redu_dim = 300;
layer = 'feature';
lfw_list = '/home/lvjiangjing/Face-Aug/test_data/lfw_affine.txt';
fea_path = {'/home/lvjiangjing/Face-Aug/test_code/featues/affine_p.h5'};


pairs = importdata('pairs.mat');
people = importdata('people.mat');


%%
true_label = [ones(1,300) zeros(1,300)]';
total_label = repmat(true_label,1,10);
total_socre = zeros(600,10);
fid = fopen([savepath,'/results/recorate.txt'],'w+');
acc = zeros(10,1);
temppara = [1e-2,1e-3,1e-4,1e-5,1e-6];
num = length(temppara);
for k=1:10
    fprintf(fid,'Set %d: \r\n',k);
    fprintf('Set %d: \r\n',k);
    [train probe_idx gallery_idx] = getdata(people ,pairs, lfw_list,k );
    
    %% train pca
    if 1
        pca_proj = pca_train(fea_path,train,redu_dim,layer);  %reduce to 1000 dimensions ;n*1000
        save([savepath,'/models/pca_proj_to1000_' num2str(k) '.mat'],'pca_proj');
    else
        pca_proj = importdata([savepath,'/models/pca_proj_to1000_' num2str(k) '.mat']);
    end
    
    fea = getpcadata(fea_path,pca_proj,layer);
    probe = fea(:,probe_idx);
    gallery = fea(:,gallery_idx);
    %% train plda
    if 1
        label.id = train.label;
        train_fea= fea(:,train.idx);
        para.d = [];
        para.l1 = temppara(2);
        para.l2 = temppara(2);
        [plda_mdl,~] = plda_train_simple(train_fea,label.id,para);
        save([savepath,'/models/mdl' num2str(k) '.mat'],'plda_mdl');
    else
        plda_mdl = importdata([savepath,'/models/mdl' num2str(k) '.mat']);
    end
    %% get score
    tempscore = plda_eval(plda_mdl,probe, gallery);
    score = sigmf(tempscore,[0.01,0]);
    
    %% draw ROC
    h = figure(k);
    [acc(k),~,thres] = drawROC(diag(score), true_label);
    saveas(h,[savepath,'/results/hg_roc_' num2str(k) '.jpg']);
    close(h);
    
    %     pred_label = (diag(score)>thres);
    %     err_savepath = [savepath,'/err_pairs'];
    %     error_pairs(pairs,pred_label,true_label,diag(score),err_savepath,k);
    
    total_socre(:,k) = diag(score)-thres;
    if 1
        pred_label = (total_socre(:,k)>0);
        err_savepath = [savepath,'/err_pairs'];
        error_pairs(pred_label,true_label,total_socre(:,k),err_savepath,k);
    end
    
    
    fprintf(fid,'Acc = %.4f\r\n',acc(k));
    fprintf('Acc = %.4f\r\n',acc(k));
    
end
h = figure(11);
drawROC(total_socre(:), total_label);
saveas(h,[savepath,'/results/average_roc.jpg']);
close(h);
score = total_socre(:);
label = total_label;
save([savepath,'/results/average_roc.mat'],'score','label');

acc_mean = mean(acc);
acc_var = std(acc);
fprintf(fid,'\r\n Acc_mean = %.4f, Acc_var = %.4f \r\n',acc_mean ,acc_var);
fprintf('\r\n Acc_mean = %.4f, Acc_var = %.4f \r\n',acc_mean ,acc_var);
acc_mean ,acc_var
fclose(fid);
