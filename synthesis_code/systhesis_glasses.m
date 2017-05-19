clear
clc
scrTri = [177 208; 269 208; 223 287];
% read mask info
glass = imread('../norm/glasses/glass0.jpg');
glass_alpha = imread('../norm/glasses/glass0_alpha.jpg');
% read img info
pts = importdata('images/296961468_1.pts');
img = imread('images/296961468_1.jpg');
img = mat2gray(img);
%get transform matrix
dstTri = round([mean(pts(37:42,:),1);mean(pts(43:48,:),1);mean(pts(49:60,:),1)]);
Tfm =  cp2tform(scrTri, dstTri, 'similarity');

% transform glasses mask
imgSize = size(img);
Tfm_glass = imtransform(glass, Tfm, 'XData', [1 imgSize(2)],...
                                      'YData', [1 imgSize(1)], 'Size', imgSize);
Tfm_glass_alpha = imtransform(glass_alpha, Tfm, 'XData', [1 imgSize(2)],...
                                      'YData', [1 imgSize(1)], 'Size', imgSize(1:2));
Tfm_glass_alpha = mat2gray(Tfm_glass_alpha);                                  
Tfm_glass = mat2gray(Tfm_glass);
% merge img
outImg = bsxfun(@times, img, 1-Tfm_glass_alpha) + bsxfun(@times, Tfm_glass, Tfm_glass_alpha);
imshow(outImg);