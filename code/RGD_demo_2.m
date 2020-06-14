function [ nbeta, sbeta, obj] = RGD_demo_2( M, maxiter)
%RGD_DEMO 
%   To solve min_\beta \beta^t M \beta s.t. \sum \beta = 1, \beta >= 0 via
%   reduced gradient descent algorithm

alpha = 1e-8;

m = size(M, 1);
beta = ones(m,1)/m;
sbeta = cell(1);
t = 1;
sbeta{t} = beta;
obj(t) = beta'*M*beta;

flag = 1;
while flag==1
    
    delta = 2*M*beta;
    
    fix_indx = zeros(m,1);
    for p=1:m
       % for convex function: delta(p)>0
       if beta(p)<=eps && delta(p)>0
           fix_indx(p) = 1;
       end
    end
    fix_indx = boolean(fix_indx);
    fix_sum = sum(beta(fix_indx));
    beta(fix_indx) = 0;
    delta(fix_indx) = 0;
    
    var_indx = ~fix_indx;
    % beta values on fix_indx sometimes are negative, for relatively bigger
    % step size. spare them to the var_indx.
    beta(var_indx) = beta(var_indx) + fix_sum/sum(var_indx);

    u = find(beta==max(beta));
    u = u(1);
    
    d = zeros(m,1);
    d(var_indx) = -(delta(var_indx) - delta(u));
    d(u) = -(sum(var_indx)*delta(u) - sum(delta(var_indx)));
    
    nbeta = beta + alpha*d;
    if max(abs(nbeta - beta)) < 1e-6 || t>maxiter

        flag = 0;
    end

    t = t+1;
    beta = nbeta;
    sbeta{t} = nbeta;
    obj(t) = beta'*M*beta;
    

    
end 

end

