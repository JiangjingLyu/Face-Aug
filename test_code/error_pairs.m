% function error_pairs(testdata,score,pred_label,true_label,err_savepath)
function error_pairs(pred_label,true_label,score,err_savepath,k)
if ~exist(err_savepath,'dir')
    mkdir(err_savepath);
end

pairs = importdata('pairs.mat');
source_path = '/home/lvjiangjing/Face_BaseLine/LFW/Data/affine/';
% error_value
err_idx = find(pred_label~=true_label);
err_preid = pred_label(err_idx);
err_trueid = true_label(err_idx);
err_maxscore = score(err_idx);
n = length(err_idx);
bg = (k-1)*600;
for j =1:n
    h = figure(j);
    set(h,'Visible','off');
    hold on
    i = err_idx(j);
    name = [source_path pairs.person{bg+i,1} num2str(pairs.faceid(bg+i,1),'_%04d.jpg')];
    subplot(1,2,1);imshow(name);
    xlabel([pairs.person{bg+i,1} num2str(pairs.faceid(bg+i,1),'_%04d.jpg')],'FontSize',13);
    name = [source_path pairs.person{bg+i,2} num2str(pairs.faceid(bg+i,2),'_%04d.jpg')];
    subplot(1,2,2);imshow(name);
    xlabel([pairs.person{bg+i,2} num2str(pairs.faceid(bg+i,2),'_%04d.jpg')],'FontSize',13);
    text(-100,-40,sprintf('pre label:%d, true label:%d, score: %.2f',err_preid(j),err_trueid(j),err_maxscore(j)));
    saveas(h,sprintf('%s/err_%d_%d.jpg',err_savepath,k,j));
    close(h);
end
