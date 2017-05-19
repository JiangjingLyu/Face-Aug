function  [J linkp]= faceAlign_linear(img_fn, inputP,staP)
imageSize = 250;
% I = imread(img_fn);
Img = imread(img_fn);
% Img = rgb2gray(Img);

if ndims(Img)==2
    I = Img;
    [J linkp] = grayMapping(I,inputP,staP,imageSize);
else
    if ndims(Img)==3
        J = zeros(imageSize,imageSize,3,'uint8');
        for n = 1:3
            I = Img(:,:,n);
            [J(:,:,n), linkp] = grayMapping(I,inputP,staP,imageSize);
        end
    end
end
linkp = round(linkp);
end




function [J linkp]= grayMapping(I,inputP,staP,imageSize)

J = zeros(imageSize);

xy = [repmat(1:imageSize,1,imageSize);reshape(repmat(1:imageSize,imageSize,1),1,numel(J))]';
P = round([mean(inputP(37:42,:),1);mean(inputP(43:48,:),1)]); %the input face image keypoint: left eye and right eye
tform = cp2tform(P, staP, 'Linear conformal');
invT = tform.tdata.Tinv;% the inv_matrix for mapping
linkp = [inputP,ones(size(inputP,1),1)]*tform.tdata.T;
linkp = linkp(:,1:2);
xy_map = [xy,ones(numel(J),1)];  
xys = xy_map*invT;

[I1,I2] = size(I);
idx1 = find(xys(:,1)<1|xys(:,1)>I2);
idx2 = find(xys(:,2)<1|xys(:,2)>I1);
idx = union(idx1,idx2);
J(xy_map(idx,2),xy_map(idx,1)) = 0;
idxc = setdiff([1:size(xy_map,1)],idx);
for i = 1:length(idxc)
    k = idxc(i);
    xb = 1-(xys(k,2) - floor(xys(k,2)));
    xf = 1-xb;
    yb = 1-(xys(k,1) - floor(xys(k,1)));
    yf = 1-yb;
    x = floor(xys(k,2));
    y = floor(xys(k,1));
    if x > I1-1;
        x = I1-1;
    end
    if y > I2-1;
        y = I2-1;
    end
    pix1 = I(x,y)*xb+I(x+1,y)*xf;
    pix2 = I(x,y+1)*xb+I(x+1,y+1)*xf;
    pix = pix1*yb+pix2*yf;
  
    J(xy_map(k,2),xy_map(k,1)) = pix;
end
J = uint8(J);
end