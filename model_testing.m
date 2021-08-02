function [score] = model_testing( fbfeat,  pentropyTrain, instfreqTrain, scattTrain,classes,model)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
node2_data=[pentropyTrain; instfreqTrain];
scatt_feat=feature_mean(scattTrain);
num_classes = length(classes);
score = zeros([1,num_classes]);
for j=1:size(classes,2)
    classes_double(j)=str2double(classes{j});
end
scored_labels=[164889003	164890007	6374002	426627000	733534002 164909002	713427006 59118001	270492004	713426002	39732003	445118002	164947007	251146004	111975006	698252002	426783006	284470004 63593006	10370003	365413008	427172004 17338001	164917005	47665007	427393009	426177001	427084000	164934002	59931005];
for j=1:size(scored_labels,2)
    scored_label_idx(j)=find(classes_double==scored_labels(j));
end
[prednode1 ,scores_node1] = classify(model{1},scattTrain);
if prednode1=='1'
    score(17)= scores_node1(1,1);
else
[prednode2 ,scores_node2] = classify(model{2},node2_data);
for i=1:size(scatt_feat,1)
    [prednew_af_afl, scores_af_afl]=predict(model{3}{i},scatt_feat(i,:));
    score_af_afl1{i}=scores_af_afl./sum(scores_af_afl,2);
    [prednew_afvsafl, scores_afvsafl]=predict(model{4}{i},scatt_feat(i,:));
    score_afvsafl1{i}=scores_afvsafl./sum(scores_afvsafl,2);
    [prednew_brady, scores_brady]=predict(model{5}{i},scatt_feat(i,:));
    score_brady1{i}=scores_brady./sum(scores_brady,2);
    [prednew_lpr, scores_lpr]=predict(model{6}{i},scatt_feat(i,:));
    score_lpr1{i}=scores_lpr./sum(scores_lpr,2);
    [prednew_lqt, scores_lqt]=predict(model{7}{i},scatt_feat(i,:));
    score_lqt1{i}=scores_lqt./sum(scores_lqt,2);
    [prednew_pac, scores_pac]=predict(model{8}{i},scatt_feat(i,:));
    score_pac1{i}=scores_pac./sum(scores_pac,2);
    [prednew_pvc, scores_pvc]=predict(model{9}{i},scatt_feat(i,:));
    score_pvc1{i}=scores_pvc./sum(scores_pvc,2);
    [prednew_sb, scores_sb]=predict(model{10}{i},scatt_feat(i,:));
    score_sb1{i}=scores_sb./sum(scores_sb,2);
    [prednew_stach, scores_stach]=predict(model{11}{i},scatt_feat(i,:));
    score_stach1{i}=scores_stach./sum(scores_stach,2);
    [prednew_sa, scores_sa]=predict(model{12}{i},scatt_feat(i,:));
    score_sa1{i}=scores_sa./sum(scores_sa,2);
    [prednew_allbbb, scores_allbbb]=predict(model{13}{i},scatt_feat(i,:));
    score_allbbb1{i}=scores_allbbb./sum(scores_allbbb,2);
    [prednew_crbbb, scores_crbbb]=predict(model{14}{i},scatt_feat(i,:));
    score_crbbb1{i}=scores_crbbb./sum(scores_crbbb,2);
    [prednew_clbbb, scores_clbbb]=predict(model{15}{i},scatt_feat(i,:));
    score_clbbb1{i}=scores_clbbb./sum(scores_clbbb,2);
    [prednew_bbb, scores_bbb]=predict(model{16}{i},scatt_feat(i,:));
    score_bbb1{i}=scores_bbb./sum(scores_bbb,2);
    [prednew_irbbb, scores_irbbb]=predict(model{17}{i},scatt_feat(i,:));
    score_irbbb1{i}=scores_irbbb./sum(scores_irbbb,2);
    [prednew_iavb, scores_iavb]=predict(model{18}{i},scatt_feat(i,:));
    score_iavb1{i}=scores_iavb./sum(scores_iavb,2);
    [prednew_lad, scores_lad]=predict(model{19}{i},scatt_feat(i,:));
    score_lad1{i}=scores_lad./sum(scores_lad,2);
    [prednew_lanfb, scores_lanfb]=predict(model{20}{i},scatt_feat(i,:));
    score_lanfb1{i}=scores_lanfb./sum(scores_lanfb,2);
    [prednew_lqrsv, scores_lqrsv]=predict(model{21}{i},scatt_feat(i,:));
    score_lqrsv1{i}=scores_lqrsv./sum(scores_lqrsv,2);
    [prednew_nsivcb, scores_nsivcb]=predict(model{22}{i},scatt_feat(i,:));
    score_nsivcb1{i}=scores_nsivcb./sum(scores_nsivcb,2);
    [prednew_pr, scores_pr]=predict(model{23}{i},scatt_feat(i,:));
    score_pr1{i}=scores_pr./sum(scores_pr,2);
    [prednew_prwp, scores_prwp]=predict(model{24}{i},scatt_feat(i,:));
    score_prwp1{i}=scores_prwp./sum(scores_prwp,2);
    [prednew_qab, scores_qab]=predict(model{25}{i},scatt_feat(i,:));
    score_qab1{i}=scores_qab./sum(scores_qab,2);
    [prednew_rad, scores_rad]=predict(model{26}{i},scatt_feat(i,:));
    score_rad1{i}=scores_rad./sum(scores_rad,2);
    [prednew_tab, scores_tab]=predict(model{27}{i},scatt_feat(i,:));
    score_tab1{i}=scores_tab./sum(scores_tab,2);
    [prednew_tinv, scores_tinv]=predict(model{28}{i},scatt_feat(i,:));
    score_tinv1{i}=scores_tinv./sum(scores_tinv,2);
end
fscore_af_afl=ones(length(prednew_af_afl),2);
fscore_afvsafl=ones(length(prednew_afvsafl),2);
fscore_brady=ones(length(prednew_brady),2);
fscore_lpr=ones(length(prednew_lpr),2);
fscore_lqt=ones(length(prednew_lqt),2);
fscore_pac=ones(length(prednew_pac),2);
fscore_pvc=ones(length(prednew_pvc),2);
fscore_sb=ones(length(prednew_sb),2);
fscore_stach=ones(length(prednew_stach),2);
fscore_sa=ones(length(prednew_sa),2);
fscore_allbbb=ones(length(prednew_allbbb),2);
fscore_crbbb=ones(length(prednew_crbbb),2);
fscore_clbbb=ones(length(prednew_clbbb),2);
fscore_bbb=ones(length(prednew_bbb),2);
fscore_irbbb=ones(length(prednew_irbbb),2);
fscore_iavb=ones(length(prednew_iavb),2);
fscore_lad=ones(length(prednew_lad),2);
fscore_lanfb=ones(length(prednew_lanfb),2);
fscore_lqrsv=ones(length(prednew_lqrsv),2);
fscore_nsivcb=ones(length(prednew_nsivcb),2);
fscore_pr=ones(length(prednew_pr),2);
fscore_prwp=ones(length(prednew_prwp),2);
fscore_qab=ones(length(prednew_qab),2);
fscore_rad=ones(length(prednew_rad),2);
fscore_tab=ones(length(prednew_tab),2);
fscore_tinv=ones(length(prednew_tinv),2);
for i=1:size(score_af_afl1,2)
    fscore_af_afl=fscore_af_afl.*score_af_afl1{i};
    fscore_afvsafl=fscore_afvsafl.*score_afvsafl1{i};
    fscore_brady=fscore_brady.*score_brady1{i};
    fscore_lpr=fscore_lpr.*score_lpr1{i};
    fscore_pac=fscore_pac.*score_pac1{i};
    fscore_lqt=fscore_lqt.*score_lqt1{i};
    fscore_pvc=fscore_pvc.*score_pvc1{i};
    fscore_sb=fscore_sb.*score_sb1{i};
    fscore_stach=fscore_stach.*score_stach1{i};
    fscore_sa=fscore_sa.*score_sa1{i};
    fscore_allbbb=fscore_allbbb.*score_allbbb1{i};
    fscore_crbbb=fscore_crbbb.*score_crbbb1{i};
    fscore_clbbb=fscore_clbbb.*score_clbbb1{i};
    fscore_bbb=fscore_bbb.*score_bbb1{i};
    fscore_irbbb=fscore_irbbb.*score_irbbb1{i};
    fscore_iavb=fscore_iavb.*score_iavb1{i};
    fscore_lad=fscore_lad.*score_lad1{i};
    fscore_lanfb=fscore_lanfb.*score_lanfb1{i};
    fscore_lqrsv=fscore_lqrsv.*score_lqrsv1{i};
    fscore_nsivcb=fscore_nsivcb.*score_nsivcb1{i};
    fscore_pr=fscore_pr.*score_pr1{i};
    fscore_prwp=fscore_prwp.*score_prwp1{i};
    fscore_qab=fscore_qab.*score_qab1{i};
    fscore_rad=fscore_rad.*score_rad1{i};
    fscore_tab=fscore_tab.*score_tab1{i};
    fscore_tinv=fscore_tinv.*score_tinv1{i};
end
score=zeros(1,length(classes));
score(scored_label_idx([1:16 18:30]))=[fscore_afvsafl(1,2) fscore_afvsafl(1,1) fscore_bbb(1,2) fscore_brady(1,2) fscore_clbbb(1,2) fscore_clbbb(1,2) fscore_crbbb(1,2) fscore_crbbb(1,2) fscore_iavb(1,2) fscore_irbbb(1,2) fscore_lad(1,2) fscore_lanfb(1,2) fscore_lpr(1,2) fscore_lqrsv(1,2) fscore_lqt(1,2)...
    fscore_nsivcb(1,2) fscore_pac(1,2)  fscore_pac(1,2)  fscore_pr(1,2)  fscore_prwp(1,2)   fscore_pvc(1,2)  fscore_pvc(1,2)  fscore_qab(1,2)  fscore_rad(1,2)  fscore_sa(1,2)  fscore_sb(1,2)  fscore_stach(1,2)  fscore_tab(1,2)  fscore_tinv(1,2)];
end
end

