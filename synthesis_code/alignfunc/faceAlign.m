function  J = faceAlign(img_fn, P,staKey)
imageSize = 160;
I = imread(img_fn);
if ndims(I) == 3
    I = rgb2gray(I);
end
PM = staKey;%the standard face image keypoint

tform = cp2tform(reshape(P, 2, [])', reshape(PM, 2, [])', 'affine');
invT = tform.tdata.Tinv;
xy = [repmat([1:160],1,160);reshape(repmat([1:160],160,1),1,160*160);ones(1,160*160)]';
xys = xy*invT;

J = zeros(imageSize,imageSize);
pix = zeros(imageSize*imageSize,1);
for k = 1:imageSize*imageSize
    if xys(k,1)<1 || xys(k,2)<1||xys(k,1)>size(I,2)||xys(k,2)>size(I,1)
        pix(k) = 0;
    else
        xb = 1-(xys(k,1) - floor(xys(k,1)));
        xf = 1-xb;
        yb = 1-(xys(k,2) - floor(xys(k,2)));
        yf = 1-yb;
        x = floor(xys(k,2));
        y = floor(xys(k,1));
        pix1 = I(x,y)*xb+I(x+1,y)*xf;
        pix2 = I(x,y+1)*xb+I(x+1,y+1)*xf;
        pix(k) = pix1*yb+pix2*yf;
%         pix(k) = I(floor(xys(k,2)),floor(xys(k,1)))*(xb+yb)/4+...
%             I(floor(xys(k,2)),ceil(xys(k,1)))*(xb+yf)/4+...
%             I(ceil(xys(k,2)),floor(xys(k,1)))*(xf+yb)/4+...
%             I(ceil(xys(k,2)),ceil(xys(k,1)))*(xf+yf)/4;
%          pix(k) = I(round(xys(k,2)),round(xys(k,1)));
    end
end
J = uint8(reshape(pix,imageSize,imageSize))';

