function testdata = gettestset(fea,label,k)

rand('seed',k);% rand seed set 
                             % gallery %
per_num = 1; %gallery number pic per id;
[probe_idx,gallery_idx] = crossvalind('LeaveMOut',label.id,per_num);

gallery.fea = fea(:,gallery_idx);
gallery.id = label.id(gallery_idx);
gallery.ids = gallery.id;
gallery.lists = label.list(gallery_idx);
[gallery.fea,gallery.id] = FindCentroid(gallery.fea,gallery.id);
gallery.fea = normc_safe(gallery.fea);
                              % probe %
probe.fea = fea(:,probe_idx);
probe.id = label.id(probe_idx);
probe.list = label.list(probe_idx);

testdata.gallery = gallery;
testdata.probe = probe;
