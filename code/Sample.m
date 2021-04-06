function [Sample_pos, rest_pos] = Sample(Apos,k)

pos =  randperm(length(Apos));
Sample_pos = pos(1:k)';
rest_pos = setdiff(Apos, Sample_pos);