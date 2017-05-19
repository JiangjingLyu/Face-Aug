function aimKp = kp_add(s_pts)

rct = [min(s_pts(:,1)) min(s_pts(:,2)) max(s_pts(:,1)) max(s_pts(:,2))];
w = rct(3)-rct(1)+1;
h = rct(4)-rct(2)+1;
scale = 0.3;
rct(1) = round(rct(1)-scale*w);
rct(2) = round(rct(2)-0.8*h);
rct(3) = round(rct(3)+scale*w);
rct(4) = round(rct(4)+0.4*h);

mid_w = round((rct(1)+rct(3))/2);
mid_h = round((rct(2)+rct(4))/2);
addkp = [rct(1),rct(2); rct(1),mid_h; rct(1), rct(4); mid_w, rct(4); rct(3),rct(4); rct(3), mid_h; rct(3),rct(2); mid_w, rct(2)];
aimKp = [s_pts;addkp];
end

