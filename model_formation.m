function [model] = model_formation(fb_feat, pentropyfeat_leads, instfreqfeat_leads, scatt_feat_leads,single_lab)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
classes2 = ["1" "2"];
%qab_lab=nonbbb_labx(:,20);
node1_data=scatt_feat_leads;
node1_lab=single_lab(:,17);
node1_lab(node1_lab==0)=2;
YTrain=categorical(node1_lab);
layers = [ ...
sequenceInputLayer(size(node1_data{1,1},1))
bilstmLayer(100,'OutputMode','last')
fullyConnectedLayer(2)
softmaxLayer
classificationLayer
];
options = trainingOptions('adam', ...
'MaxEpochs',300, ...
'MiniBatchSize', 500, ...
'InitialLearnRate', 0.01, ...
'GradientThreshold', 1, ...
'ExecutionEnvironment',"auto",...
'plots','training-progress', ...
'Verbose',false);
netnode1 = trainNetwork(node1_data,YTrain,layers,options);
model{1}=netnode1;

node2_idx=find(node1_lab==2);
peninf=cellfun(@(x, y) [x; y], pentropyfeat_leads, instfreqfeat_leads,'UniformOutput',false);
node2_sct1=node1_data(node2_idx);
node2_sct = cellfun(@(x) feature_mean(x),node2_sct1,'UniformOutput',false);
node2_data=peninf(node2_idx);
node2_labx=single_lab(node2_idx,:);
ryt_idx=find(node2_labx(:,1)==1 | node2_labx(:,2)==1 | node2_labx(:,4)==1 | node2_labx(:,13)==1 | node2_labx(:,15)==1 | node2_labx(:,18)==1 | node2_labx(:,19)==1 | node2_labx(:,22)==1 | node2_labx(:,23)==1);%column 15 is normal
node2_lab=zeros(length(node2_idx),1);
node2_lab(ryt_idx)=1;
YTrain2=node2_lab;
YTrain2(YTrain2==0)=2;
YTrain2=categorical(YTrain2);
classWeights = 1./countcats(YTrain2);
classWeights = classWeights'/mean(classWeights);
layers2 = [ ...
sequenceInputLayer(size(node2_data{1,1},1))
lstmLayer(100,'OutputMode','last','InputWeightsInitializer','he')
fullyConnectedLayer(2,'WeightsInitializer','he')
softmaxLayer
classificationLayer('Classes',classes2,'ClassWeights',classWeights)];

options2 = trainingOptions('adam', ...
'MaxEpochs',300, ...
'MiniBatchSize', 1500, ...
'InitialLearnRate', 0.01, ...
'GradientThreshold', 1, ...
'ExecutionEnvironment',"auto",...
'plots','training-progress', ...
'Verbose',false);
netnode2 = trainNetwork(node2_data,YTrain2,layers2,options2);
model{2}=netnode2;

rythm_branch_idx=find(node2_lab==1);
rytreelabels=node2_labx(rythm_branch_idx,:);
rytdata=node2_sct(rythm_branch_idx);
af_afl_lab=rytreelabels(:,1)+rytreelabels(:,2);
af_afl_lab(find(af_afl_lab>1))=1;
for i=1:size(rytdata{1},1)
    traind = cellfun(@(x) x(i,:),rytdata,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=af_afl_lab;
    t = ClassificationTree.template('minleaf',1);
    model_af_afl{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{3}=model_af_afl;

afvsafl_idx=find(af_afl_lab==1);
afvsafl_labx=rytreelabels(afvsafl_idx,:);
afvsafl_lab=afvsafl_labx(:,1);
afvsafl_data=rytdata(afvsafl_idx);
for i=1:size(afvsafl_data{1},1)
    traind = cellfun(@(x) x(i,:),afvsafl_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=afvsafl_lab;
    t = ClassificationTree.template('minleaf',1);
    model_afvsafl{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{4}=model_afvsafl;

other_rythms_idx=find(af_afl_lab==0);
other_rythms_labx=rytreelabels(other_rythms_idx,:);
brady_lab=other_rythms_labx(:,4);
other_rythms_data=rytdata(other_rythms_idx);
for i=1:size(other_rythms_data{1},1)
    traind = cellfun(@(x) x(i,:),other_rythms_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=brady_lab;
    t = ClassificationTree.template('minleaf',1);
    model_brady{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{5}=model_brady;

lpr_lab=other_rythms_labx(:,13);
for i=1:size(other_rythms_data{1},1)
    traind = cellfun(@(x) x(i,:),other_rythms_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=lpr_lab;
    t = ClassificationTree.template('minleaf',1);
    model_lpr{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{6}=model_lpr;

lqt_lab=other_rythms_labx(:,15);
for i=1:size(other_rythms_data{1},1)
    traind = cellfun(@(x) x(i,:),other_rythms_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=lqt_lab;
    t = ClassificationTree.template('minleaf',1);
    model_lqt{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{7}=model_lqt;

pac_lab=other_rythms_labx(:,18) + other_rythms_labx(:,19);
pac_lab(find(pac_lab>1))=1;
for i=1:size(other_rythms_data{1},1)
    traind = cellfun(@(x) x(i,:),other_rythms_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=pac_lab;
    t = ClassificationTree.template('minleaf',1);
    model_pac{i} = fitensemble(traindata,Ytrain,'Bag',90,t,'type','classification');
end
model{8}=model_pac;


pvc_lab=other_rythms_labx(:,22) + other_rythms_labx(:,23);
pvc_lab(find(pvc_lab>1))=1;
for i=1:size(other_rythms_data{1},1)
    traind = cellfun(@(x) x(i,:),other_rythms_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=pvc_lab;
    t = ClassificationTree.template('minleaf',1);
    model_pvc{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{9}=model_pvc;

other_branch_idx=find(node2_lab==0);
othertreelabels=node2_labx(other_branch_idx,:);
other_branch_data=node2_sct(other_branch_idx);
sb_lab=othertreelabels(:,27);
for i=1:size(other_branch_data{1},1)
    traind = cellfun(@(x) x(i,:),other_branch_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=sb_lab;
    t = ClassificationTree.template('minleaf',1);
    model_sb{i} = fitensemble(traindata,Ytrain,'Bag',90,t,'type','classification');
end
model{10}=model_sb;

nonsb_idx=find(sb_lab==0);
nonsb_labx=othertreelabels(nonsb_idx,:);
stach_lab=nonsb_labx(:,28);
stach_data=other_branch_data(nonsb_idx);
for i=1:size(stach_data{1},1)
    traind = cellfun(@(x) x(i,:),stach_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=stach_lab;
    t = ClassificationTree.template('minleaf',1);
    model_stach{i} = fitensemble(traindata,Ytrain,'Bag',90,t,'type','classification');
end
model{11}=model_stach;

nonst_idx=find(stach_lab==0);
sa_data=stach_data(nonst_idx);
nonst_labx=nonsb_labx(nonst_idx,:);
sa_lab=nonst_labx(:,26);
for i=1:size(sa_data{1},1)
    traind = cellfun(@(x) x(i,:),sa_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=sa_lab;
    t = ClassificationTree.template('minleaf',1);
    model_sa{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{12}=model_sa;

nonsa_idx=find(sa_lab==0);
nonsa_data=sa_data(nonsa_idx);
nonsa_labx=nonst_labx(nonsa_idx,:);
bbbnode_lab=nonsa_labx(:,3)+nonsa_labx(:,5)+nonsa_labx(:,6)+nonsa_labx(:,7)+nonsa_labx(:,8)+nonsa_labx(:,10);
bbbnode_lab(find(bbbnode_lab>1))=1;
for i=1:size(nonsa_data{1},1)
    traind = cellfun(@(x) x(i,:),nonsa_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=bbbnode_lab;
    t = ClassificationTree.template('minleaf',1);
    model_allbbb{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{13}=model_allbbb;

bbb_idx=find(bbbnode_lab==1);
allbbb_data=nonsa_data(bbb_idx);
bbb_labx=nonsa_labx(bbb_idx,:);
crbbb_lab=bbb_labx(:,7)+ bbb_labx(:,8);
crbbb_lab(find(crbbb_lab>1))=1;
for i=1:size(allbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),allbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=crbbb_lab;
    t = ClassificationTree.template('minleaf',1);
    model_crbbb{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{14}=model_crbbb;

clbbb_lab=bbb_labx(:,5)+ bbb_labx(:,6);
clbbb_lab(find(clbbb_lab>1))=1;
for i=1:size(allbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),allbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=clbbb_lab;
    t = ClassificationTree.template('minleaf',1);
    model_clbbb{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{15}=model_clbbb;

bbb_lab=bbb_labx(:,3);
for i=1:size(allbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),allbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=bbb_lab;
    t = ClassificationTree.template('minleaf',1);
    model_bbb{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{16}=model_bbb;

irbbb_lab=bbb_labx(:,10);
for i=1:size(allbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),allbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=irbbb_lab;
    t = ClassificationTree.template('minleaf',1);
    model_irbbb{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{17}=model_irbbb;


nonbbb_idx=find(bbbnode_lab==0);
nonbbb_data=nonsa_data(nonbbb_idx);
nonbbb_labx=nonsa_labx(nonbbb_idx,:);
iavb_lab=nonbbb_labx(:,9);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=iavb_lab;
    t = ClassificationTree.template('minleaf',1);
    model_iavb{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{18}=model_iavb;


lad_lab=nonbbb_labx(:,11);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=lad_lab;
    t = ClassificationTree.template('minleaf',1);
    model_lad{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{19}=model_lad;

lanfb_lab=nonbbb_labx(:,12);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=lanfb_lab;
    t = ClassificationTree.template('minleaf',1);
    model_lanfb{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{20}=model_lanfb;

lqrsv_lab=nonbbb_labx(:,14);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=lqrsv_lab;
    t = ClassificationTree.template('minleaf',1);
    model_lqrsv{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{21}=model_lqrsv;

nsivcb_lab=nonbbb_labx(:,16);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=nsivcb_lab;
    t = ClassificationTree.template('minleaf',1);
    model_nsivcb{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{22}=model_nsivcb;

pr_lab=nonbbb_labx(:,20);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=pr_lab;
    t = ClassificationTree.template('minleaf',1);
    model_pr{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{23}=model_pr;

prwp_lab=nonbbb_labx(:,21);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=prwp_lab;
    t = ClassificationTree.template('minleaf',1);
    model_prwp{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{24}=model_prwp;

qab_lab=nonbbb_labx(:,24);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=qab_lab;
    t = ClassificationTree.template('minleaf',1);
    model_qab{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{25}=model_qab;

rad_lab=nonbbb_labx(:,25);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=rad_lab;
    t = ClassificationTree.template('minleaf',1);
    model_rad{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{26}=model_rad;

tab_lab=nonbbb_labx(:,29);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=tab_lab;
    t = ClassificationTree.template('minleaf',1);
    model_tab{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{27}=model_tab;

tinv_lab=nonbbb_labx(:,30);
for i=1:size(nonbbb_data{1},1)
    traind = cellfun(@(x) x(i,:),nonbbb_data,'UniformOutput',false);
    data=cat(1,traind{:});
    traindata=data;
    Ytrain=tinv_lab;
    t = ClassificationTree.template('minleaf',1);
    model_tinv{i} = fitensemble(traindata,Ytrain,'RusBoost',90,t,'type','classification');
end
model{28}=model_tinv;

end

