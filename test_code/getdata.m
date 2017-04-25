function [train probe gallery] = getdata( people ,pairs, list_path,k )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%% pre set data
m = 2;
st = (k-1)*600+1;
ed = k*600;
list = textread(list_path,'%s');
list_name = regexp(list,'\w+(?=\.)','match','once');

test_id = unique(pairs.id(st:ed,:));
train_id = people.id;
train_id = setdiff(train_id,test_id);
num_train_id = people.maxface(train_id);
train_id = train_id(find(num_train_id>=m));
%% get probe and gallery
probe_feaid = zeros(600,1);
gallery_feaid = zeros(600,1);
for i=st:ed
    name = [pairs.person{i,1} num2str(pairs.faceid(i,1),'_%04d')];
    probe_feaid(i-st+1) = find(strcmp(list_name,name));
    name = [pairs.person{i,2} num2str(pairs.faceid(i,2),'_%04d')];
    gallery_feaid(i-st+1) = find(strcmp(list_name,name));
end

%% get train plda data
label = [];
train_feaid = zeros(sum(people.maxface),1);
count = 1;
n = length(train_id);
for i=1:n
    face_num = people.maxface(train_id(i));
    label = [label;train_id(i)*ones(face_num,1)];
    for j=1:face_num
        name = [people.person{train_id(i)} num2str(j,'_%04d')];
        train_feaid(count) = find(strcmp(list_name,name));
        count = count+1;
    end
end

probe = probe_feaid;
gallery = gallery_feaid;
train.idx = train_feaid(1:count-1);
train.label = label;
end

