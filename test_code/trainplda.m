function plda_mdl = trainplda(tfea,label)

train.fea = normc_safe(tfea);
train.id = label.id;

%% train plda model
para.d = [];        % subspace dimentions (default or [] to keep all)
para.l1 = 1e-4;    % smooth on within-class cov
para.l2 = 1e-4;    % smooth on between-class cov
% train the model
[plda_mdl, ~] = plda_train_simple(train.fea,train.id, para);
