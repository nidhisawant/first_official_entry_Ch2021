function xx2 = scattering_features_arrange(data,leads_idx)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
xx2=[];
for i=1:length(leads_idx) 
    xx2=[xx2; data((leads_idx(i)-1)*10+1:leads_idx(i)*10,:)];
end

end

