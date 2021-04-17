function [Val_pos, TT, rest_pos] = Sample(Apos,k,p)

% Random sampling
% Apos: All position set
% k: Number of samples
pos =  randperm(length(Apos));
Val_pos = Apos(pos(1:k)');
rest_pos = setdiff(Apos, Val_pos);
pos =  randperm(length(rest_pos));
TT = rest_pos(pos(1:p)');
rest_pos = setdiff(rest_pos, TT);