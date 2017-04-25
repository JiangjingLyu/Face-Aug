function fea = getpcadata(sourcepath,pca_proj,varargin)
disp('load the feature data ...')
if (~isempty(varargin) & ischar(varargin{1}))
    layer = varargin{1};
else
    layer = 'feature';
end
fea = [];
for i=1:length(sourcepath)
    if strcmp(sourcepath{i}(end-2:end),'bin')
        fea0 = CigitBinRead(sourcepath{i},13568);
    elseif strcmp(sourcepath{i}(end-1:end),'h5')
        
        fea0 = hdf5read(sourcepath{i},layer);
    else
        fea0 = importdata(sourcepath{i});
    end
    fea0 = normc_safe(fea0);
    fea = [fea;fea0];
end
fea = normc_safe(fea);
fea = (fea'*pca_proj)';
% fea = normc_safe(fea);
