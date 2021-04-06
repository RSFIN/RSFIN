function [Sample_pos, rest_pos] = Sample(Apos,k)

% Random sampling
% Apos: All position set
% k: Number of samples
pos =  randperm(length(Apos));
Sample_pos = pos(1:k)';
rest_pos = setdiff(Apos, Sample_pos);