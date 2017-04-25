function [acc,area,thres,points] = drawROC(output, ts_label, disp, query_flms)
% author lvjiangjing
% email: lvjiangjing12@gmail.com
% data: 2013/10/11
if ~exist('disp','var')
    disp = 1;
end

output = output(:);
ts_label = ts_label(:);

[x,idx] = sort(output);
y = ts_label(idx);
y(y~=1) = 0;   % make sure ts_label only contains 0 and 1

S = cumsum(y);
S1 = S(end) - S;
S2 = ((length(y)-1):-1:0)';

flm = (S2 - S1) / length(find(~y));     % false alarm
rcl = S1 / length(find(y));             % recall

% equal error rate
[ignore,mi] = min(flm-rcl);%min(abs(flm+rcl-1));
eer = (flm(mi(1)) + 1 - rcl(mi(1))) / 2;
acc = 1 - eer;
thres = x(mi(1));

% point of interest on the ROC curve
if exist('query_flms','var')
    if isnumeric(query_flms)
        points = zeros(length(query_flms),3);
        points(:,1) = query_flms;
        for i = 1:length(query_flms)
            [ignore,mi] = min(abs(flm-query_flms(i)));
            points(i,2) = rcl(mi(1));
            points(i,3) = x(mi(1));
        end
    elseif strcmpi(query_flms, 'all')
        points = [query_flms rcl x];
    else
        points = [];
    end
else
    points = [];
end

% area below ROC curve
idx = (flm>1e-6 & ~isnan(acc));
xx = flm(idx);
yy = rcl(idx);

% for different recalls with the same false alarm, keep the highest
% [xx,idx] = unique(xx,'first');
% yy = yy(idx);

lxx = log10(xx);
%lxx = xx;
area = sum((lxx(1:end-1)-lxx(2:end)).*(yy(1:end-1)+yy(2:end)))/2/(lxx(1)-lxx(end));

if disp  
    step = 1:floor(length(xx)/100):length(xx);
    if 1       
        plot(xx(step), yy(step), 'r');
        axis([0 1 0 1])
    else
        plot(log10(xx(step)), yy(step), 'r');
        axis([-4 0 0 1])
    end
    
    xlabel('False alarm');
    ylabel('Recall');
    title('ROC curve');
    grid on
end
