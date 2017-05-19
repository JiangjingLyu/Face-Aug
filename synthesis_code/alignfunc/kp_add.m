function aimKp = kp_add(sourceKp)
    S= 200;
    H = round(S/2);
    addkp = [1,1;1,H;1,S;H,S;S,S;S,H;S,1;H,1];
    aimKp = [sourceKp;addkp];
end