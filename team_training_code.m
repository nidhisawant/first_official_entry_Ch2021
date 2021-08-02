function  model = team_training_code(input_directory,output_directory) % train_ECG_leads_classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Train ECG leads and obtain classifier models
% for 12-lead, 6-lead, 3-lead, 4-lead and 2-lead ECG sets
% Inputs:
% 1. input_directory
% 2. output_directory
%
% Outputs:
% model: trained model
% Logistic regression models for different sets of leads
%
% Author: Erick Andres Perez Alday, PhD, <perezald@ohsu.edu>
% Version 1.0 Aug-2020
% Revision History
% By: Nadi Sadr, PhD, <nadi.sadr@dbmi.emory.edu>
% Version 2.0 1-Dec-2020
% Version 2.2 25-Jan-2021
% Version 2.3 26-April-2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define lead sets (e.g 12, 6, 4, 3 and 2 lead ECG sets)
twelve_leads = [{'I'}, {'II'}, {'III'}, {'aVR'}, {'aVL'}, {'aVF'}, {'V1'}, {'V2'}, {'V3'}, {'V4'}, {'V5'}, {'V6'}];
six_leads    = [{'I'}, {'II'}, {'III'}, {'aVR'}, {'aVL'}, {'aVF'}];
four_leads   = [{'I'}, {'II'}, {'III'}, {'V2'}];
three_leads  = [{'I'}, {'II'}, {'V2'}];
two_leads    = [{'I'}, {'II'}];
lead_sets = {twelve_leads, six_leads, four_leads, three_leads, two_leads};

disp('Loading data...')

% Find files.
input_files = {};
features =[];
for f = dir(input_directory)'
    if exist(fullfile(input_directory, f.name), 'file') == 2 && f.name(1) ~= '.' && all(f.name(end - 2 : end) == 'mat')
        input_files{end + 1} = f.name;
    end
end

% Extract classes from dataset.
% read number of unique classes
classes = get_classes(input_directory,input_files);
num_classes = length(classes);     % number of classes
num_files   = length(input_files);
Total_data  = cell(1,num_files);
Total_header= cell(1,num_files);

for j=1:size(classes,2)
    classes_double(j)=str2double(classes{j});
end
scored_labels=[164889003	164890007	6374002	426627000	733534002 164909002	713427006 59118001	270492004	713426002	39732003	445118002	164947007	251146004	111975006	698252002	426783006	284470004 63593006	10370003	365413008	427172004 17338001	164917005	47665007	427393009	426177001	427084000	164934002	59931005];
for j=1:size(scored_labels,2)
    scored_label_idx(j)=find(classes_double==scored_labels(j));
end

%% Load data recordings and header files
% Iterate over files.
disp('Training model..')

label=zeros(num_files,num_classes);
kk=1;
for i = 1:num_files
    disp(['    ', num2str(i), '/', num2str(num_files), '...'])
    % Load data.
    file_tmp = strsplit(input_files{i},'.');
    tmp_input_file = fullfile(input_directory, file_tmp{1});
    [data,header_data] = load_challenge_data(tmp_input_file);
    %% Check the number of available ECG leads
    tmp_hea = strsplit(header_data{1},' ');
    num_leads = str2num(tmp_hea{2});
    [leads, leads_idx] = get_leads(header_data,num_leads);
    
    %% Extract labels
    for j = 1 : length(header_data)
        if startsWith(header_data{j},'#Dx')
            tmp = strsplit(header_data{j},': ');
            % Extract more than one label if avialable
            tmp_c = strsplit(tmp{2},',');
            for k=1:length(tmp_c)
                idx=find(strcmp(classes,tmp_c{k}));
                label(i,idx)=1;
            end
            break
        end
    end
    new_label=[];
    for j=1:size(scored_labels,2)
        vec=label(i,scored_label_idx(j));
        new_label=[new_label vec];
    end
    
    %% Extract features
    if sum(new_label)==1 || sum(new_label)==2
        [tmp_features, fbfeat,  pentropyTrain, instfreqTrain, scattTrain] = get_features(data,header_data,leads_idx);
        features(kk,:) = tmp_features(:);
        fb_feat(kk,:)=fbfeat;
        pentropyfeat{kk}=pentropyTrain;
        instfreqfeat{kk}=instfreqTrain;
        scatt_feat{kk}=scattTrain;
        lab_new(kk,:)=new_label;
        kk=kk+1;
        
    else
        continue;
    end
   
    
end

XV_sc = [scatt_feat{:}];
XV_sc(isnan(XV_sc))=0;
XV_sc(isinf(XV_sc))=0;
mu_sc = mean(XV_sc,2);
sg_sc = std(XV_sc,[],2);
XTrainN_sc = cellfun(@(x)(x-mu_sc)./sg_sc,scatt_feat,'UniformOutput',false);
scatt_feat2 = cellfun(@(x) isnan1(x),XTrainN_sc,'UniformOutput',false);
 

XV_inst = [instfreqfeat{:}];
XV_inst(isnan(XV_inst))=0;
XV_inst(isinf(XV_inst))=0;
mu_inst = mean(XV_inst,2);
sg_inst = std(XV_inst,[],2);
XTrainN_inst = cellfun(@(x)(x-mu_inst)./sg_inst,instfreqfeat,'UniformOutput',false);
instfreqfeat2 = cellfun(@(x) isnan1(x),XTrainN_inst,'UniformOutput',false);

XV_pen = [pentropyfeat{:}];
XV_pen(isnan(XV_pen))=0;
XV_pen(isinf(XV_pen))=0;
mu_pen = mean(XV_pen,2);
sg_pen = std(XV_pen,[],2);
XTrainN_pen = cellfun(@(x)(x-mu_pen)./sg_pen,pentropyfeat,'UniformOutput',false);
pentropyfeat2 = cellfun(@(x) isnan1(x),XTrainN_pen,'UniformOutput',false);

fb_feat(isnan(fb_feat))=0;
fb_feat(isinf(fb_feat))=0;
mu_fb = mean(fb_feat);
sg_fb = std(fb_feat,[],1);
for n=1:size(fb_feat,1);
fb_feat2(n,:)=(fb_feat(n,:)-mu_fb)./sg_fb;
end

save('mu_sg.mat','mu_fb','mu_sc','mu_pen','mu_inst','sg_sc','sg_fb','sg_pen','sg_inst','-v7.3');
 

%% train logistic regression models for the lead sets
for i=1:length(lead_sets)
    % Train ECG model
    disp(['Training ',num2str(length(lead_sets{i})),'-lead ECG model...'])
    num_leads = length(lead_sets{i});
    [leads, leads_idx] = get_leads(header_data,num_leads);
     pentropyfeat_leads=cellfun(@(x) x(leads_idx,:),pentropyfeat2,'UniformOutput',false);
     instfreqfeat_leads=cellfun(@(x) x(leads_idx,:),instfreqfeat2,'UniformOutput',false);
     scatt_feat_leads=cellfun(@(x) scattering_features_arrange(x,leads_idx),scatt_feat2,'UniformOutput',false);   
    % Features = [1:12] features from 12 ECG leads + Age + Sex
%     Features_leads_idx = [leads_idx,13,14];
%     Features_leads = features(:,Features_leads_idx);
%     model = mnrfit(Features_leads,label,'model','hierarchical');
      model=model_formation(fb_feat2, pentropyfeat_leads, instfreqfeat_leads, scatt_feat_leads, lab_new);
    save_ECGleads_model(model,output_directory,classes,num_leads);
end
end

function save_ECGleads_model(model,output_directory,classes,num_leads) %save_ECG_model
% Save results.
tmp_file = [num2str(num_leads),'_lead_ecg_model.mat'];
filename = fullfile(output_directory,tmp_file);
save(filename,'model','classes','-v7.3');

disp('Done.')
end

function save_ECGleads_features(features,output_directory) %save_ECG_model
% Save results.
tmp_file = 'features.mat';
filename=fullfile(output_directory,tmp_file);
save(filename,'features');
end

% find unique number of classes
function classes = get_classes(input_directory,files)

classes={};
num_files = length(files);
k=1;
for i = 1:num_files
    g = strrep(files{i},'.mat','.hea');
    input_file = fullfile(input_directory, g);
    fid=fopen(input_file);
    tline = fgetl(fid);
    tlines = cell(0,1);
    
    while ischar(tline)
        tlines{end+1,1} = tline;
        tline = fgetl(fid);
        if startsWith(tline,'#Dx')
            tmp = strsplit(tline,': ');
            tmp_c = strsplit(tmp{2},',');
            for j=1:length(tmp_c)
                idx2 = find(strcmp(classes,tmp_c{j}));
                if isempty(idx2)
                    classes{k}=tmp_c{j};
                    k=k+1;
                end
            end
            break
        end
    end
    
    fclose(fid);
    
end
classes=sort(classes);
end

function [data,tlines] = load_challenge_data(filename)

% Opening header file
fid=fopen([filename '.hea']);

if (fid<=0)
    disp(['error in opening file ' filename]);
end

tline = fgetl(fid);
tlines = cell(0,1);
while ischar(tline)
    tlines{end+1,1} = tline;
    tline = fgetl(fid);
end
fclose(fid);

f=load([filename '.mat']);

try
    data = f.val;
catch ex
    rethrow(ex);
end

end
