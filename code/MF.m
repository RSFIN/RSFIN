function y = MF(type,x,Config,fun)

% type: Membership function category ['gaussmf', 'gbellmf', 'trapmf',
% 'sigmf', 'trimf'].
% x: Can be a vector.
% Config: Membership function parameters.
% fun: Output state: 0: Function expressions and corresponding parameters,
% others: result.

if nargin==3
    fun = 1;
end

if strcmp(type,'gaussmf')
    if fun == 0
        y = {'exp(-(x-c).^2/2/sig^2)',{'c','sig'}};
        return;
    end
    y = gaussmf(x,Config);
    return;
end

if strcmp(type,'gbellmf')
    if fun == 0
        y = {'1/(1+abs((x-c)/a).^(2*b))',{'a','b','c'}};
        return;
    end
    y = gbellmf(x,Config);
    return;
end

if strcmp(type,'trapmf')
    if fun == 0
        y = {'(x>=a&x<=b)*((x-a)./(b-a)) + (x>=b&x<=c) + (x>=c&x<=d)*((d-x)./(d-c))',{'a','b','c','d'}};
        return;
    end
    y = trapmf(x,Config);
    return;
end

if strcmp(type,'sigmf')
    if fun == 0
        y = {'1/(1+exp(-a*(x-c)))',{'a','c'}};
        return;
    end
    y = sigmf(x,Config);
    return;
end

if strcmp(type,'trimf')
    if fun == 0
        y = {'(x>=a&x<=b)*((x-a)./(b-a)) + (x>=b&x<=c)*((c-x)./(c-b))',{'a','b','c'}};
        return;
    end
    y = trimf(x,Config);
    return;
end

function y = gaussmf(x,Config)

if length(Config) ~= 2
    error()
    return
end
sig = Config(1);
c = Config(2);

y = exp(-(x-c).^2/2/sig^2);

function y = gbellmf(x,Config)

if length(Config) ~= 3
    error()
    return
end
a = Config(1);
b = Config(2);
c = Config(3);

y = 1/(1+abs((x-c)/a).^(2*b));

function y = sigmf(x,Config)

if length(Config) ~= 2
    error()
    return
end
a = Config(1);
c = Config(2);

y = 1/(1+exp(-a*(x-c)));

function y = trapmf(x,Config)

if length(Config) ~= 4
    error()
    return
end
a = Config(1);
b = Config(2);
c = Config(3);
d = Config(4);

if x <= a
    y = 0;
elseif x <= b
    y = (x-a)./(b-a);
elseif x <= c
    y = 1;
elseif x <= d
    y = (d-x)./(d-c);
else
    y = 0;
end

function y = trimf(x,Config)

if length(Config) ~= 3
    error()
    return
end
a = Config(1);
b = Config(2);
c = Config(3);

if x <= a
    y = 0;
elseif x <= b
    y = (x-a)./(b-a);
elseif x <= c
    y = (c-x)./(c-b);
else
    y = 0;
end


function []=error()
disp('参数数目有误，检查隶属度函数种类');
