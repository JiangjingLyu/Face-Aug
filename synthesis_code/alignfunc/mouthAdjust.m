function mkp = mouthAdjust(kp)
    mkp = kp;
    m_up_y = kp([51,53],2);
    m_down_y = kp([59,57],2);
    m_upin_y = kp([62,64],2);
    m_downin_y = kp([68,66],2);
    delta = mean(m_downin_y) - mean(m_upin_y);
    if delta>0
        mkp([56:60,66:68],2) = kp([56:60,66:68],2) - delta/2;
        mkp(50:54,2) = kp(50:54,2) + delta/2;
        mkp(62:64,2) = mkp([68,67,66],2);
        mkp(7:11,2) = kp(7:11,2) - delta/4;
        mkp(32:36,2) = kp(32:36,2) + delta/4;
        mkp(28:31,2) = kp(28:31,2) + delta/8;
        mkp = round(mkp);
    end
end