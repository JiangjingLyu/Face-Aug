clear
clc
addpath('alignfunc');
sta_str_idx = load('pwModel/pwModel_out.kp');
srcTri= importdata('/home/lvjiangjing/Face-Aug/systhesis_code/pwModel/standardFacePt.txt');
sta_76kp =  kp_add(srcTri);
% read mask info
hair = imread('../norm/hair_norm/hair32.jpg');
hair_alpha = imread('../norm/hair_norm/hair32_alpha.bmp');
hair = mat2gray(hair);
hair_alpha = mat2gray(hair_alpha);
% read img info
pts = importdata('images/296961468_1.pts');
img = imread('images/296961468_1.jpg');
img = mat2gray(img);
%
dst_76kp = kp_add(pts);

%%
imgSize = size(img);

Tri_hair = zeros(imgSize);
for n = 1:3
    I = hair(:,:,n);
    Tri_hair(:,:,n) = grayMapping(I,sta_76kp,dst_76kp,sta_str_idx,imgSize(1:2));
end

Tri_hair_alpha = grayMapping(hair_alpha,sta_76kp,dst_76kp,sta_str_idx,imgSize(1:2));

% merge img
outImg = bsxfun(@times, img, 1-Tri_hair_alpha) + bsxfun(@times, Tri_hair, Tri_hair_alpha);
imshow(outImg);