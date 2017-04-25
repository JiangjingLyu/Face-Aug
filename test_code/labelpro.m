function label = labelpro(listpath)

list = {};
%%%%%%%%%%%%%%%%%%%% need to modify %%%%%%%%%%%%%%%%%%%%%%%%%%
if length(listpath)>1
    for i = 1:6
        list = [list;textread(listpath{i},'%s')];
    end
else
    list = textread(listpath{1},'%s');
end

%%%%%%%%%%%%%%%%%%%%%%%%   ID process %%%%%%%%%%%%%%%%%%%%%%%%%
id_str = regexp(list,'(?<=P)\d{5}','match');
id = zeros(size(id_str));
for i = 1:length(id_str)
    id(i) = str2double(id_str{i}{1});
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
label.id = id;
label.list = list;
