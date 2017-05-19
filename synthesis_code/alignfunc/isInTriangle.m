function ret = isInTriangle(pointXY,tranglePoints)

if ~ all(size(pointXY,2) == 2)
     error('first para is noe point!')
end
if ~ all(size(tranglePoints)==[3 2]) 
     error('sescond para must be three points,[3*2] dim!');
end
k = zeros(2,3);
c = zeros(1,3);
k(:,1) = [tranglePoints(2,2)-tranglePoints(1,2);tranglePoints(1,1)-tranglePoints(2,1)];
k(:,2) = [tranglePoints(3,2)-tranglePoints(2,2);tranglePoints(2,1)-tranglePoints(3,1)];
k(:,3) = [tranglePoints(1,2)-tranglePoints(3,2);tranglePoints(3,1)-tranglePoints(1,1)];
c(1) = det(tranglePoints([1 2],:));
c(2) = det(tranglePoints([2 3],:));
c(3) = det(tranglePoints([3 1],:));

eqw = pointXY*k-repmat(c,[size(pointXY,1),1]);
ret = all(eqw>=0,2);

