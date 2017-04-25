%% pca_train
function pca_proj = pca_train(fea_path,train,varargin)

if (~isempty(varargin) & ~ischar(varargin{1}))
    needDim = varargin{1};
    if length(varargin)>1 & ischar(varargin{2})
    layer = varargin{2};
    end
else
    needDim = 1000;
    layer = 'feature';
end

fea = [];
for i=1:length(fea_path)
    if strcmp(fea_path{i}(end-2:end),'bin')
        fea0 = CigitBinRead(fea_path{i},13568);
    elseif strcmp(fea_path{i}(end-1:end),'h5')
        fea0 = hdf5read(fea_path{i},layer);
    else
        fea0 = importdata(fea_path{i});
    end
%     fea0 = normc_safe(fea0);
    fea = [fea;fea0];
end
train_fea = fea(:,train.idx);
train_fea = normc_safe(train_fea);
%% %%%%%%%%%%%%%%%%%%%  train pca %%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('train pca ...\r\nmay be half an hour or longer time');
% addpath('func');
[~,b] = pca(train_fea',needDim);% row train_fea
pca_proj = b.M;
