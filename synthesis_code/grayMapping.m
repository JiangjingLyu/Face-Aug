function J = grayMapping(I,inputP,staP,triangleIdx,imageSize)
J = zeros(imageSize);
strangleNum = size(triangleIdx,1);
for i = 1:strangleNum
    xy = [repmat([1:imageSize(2)],1,imageSize(1));reshape(repmat([1:imageSize(1)],imageSize(2),1),1,numel(J))]';
    PM = staP(triangleIdx(i,:),:);%the standard face image keypoint
    P = inputP(triangleIdx(i,:),:);%the input face image keypoint
    try
        tform = cp2tform(P, PM, 'affine');
    catch
        J = I;
        break;    
    end
    invT = tform.tdata.Tinv;% the inv_matrix for mapping
    
    ret = isInTriangle(xy,PM);
    xy_map = [xy(ret,:),ones(sum(ret),1)];  
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
end
% J = uint8(J);
end