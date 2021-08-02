%% Apply classifier model to test set

function [score, label,classes] = team_testing_code(data,header_data, loaded_model)

model   = loaded_model.model;
classes = loaded_model.classes;
load('mu_sg.mat');
num_classes = length(classes);

label = zeros([1,num_classes]);

score = ones([1,num_classes]);

% Extract features from test data
tmp_hea = strsplit(header_data{1},' ');
num_leads = str2num(tmp_hea{2});
[leads, leads_idx] = get_leads(header_data,num_leads);
%features = get_features(data,header_data,leads_idx);
[tmp_features, fbfeat,  pentropyTrain, instfreqTrain, scattTrain] = get_features(data,header_data,leads_idx);
 fb_n=(fbfeat-mu_fb)./sg_fb; 
 pen_n=(pentropyTrain-mu_pen)./sg_pen;
 inst_n=(instfreqTrain-mu_inst)./sg_inst;
 scat_n=(scattTrain-mu_sc)./sg_sc;
 scat_n(isnan(scat_n))=0;
 pen_n(isnan(pen_n))=0;
 inst_n(isnan(inst_n))=0;
% Use your classifier here to obtain a label and score for each class.
score =model_testing(fb_n, pen_n, inst_n, scat_n, classes, model);
[~,idx] = find(score>0.5);

label(idx)=1;
end
