function [ nbeta, sbeta, obj] = RGD_demo_1( M, maxiter)
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
%     u = find(beta==max(beta));
%     u = u(1);
    u=1;
    delta = 2*M*beta;

    d = -(delta - delta(u));
    d(u) = -(m*delta(u) - sum(delta));
    
    for p=1:m
       if beta(p)<=eps && delta(p)>0
           d(p) = 0;
       end
    end
    
    nbeta = beta + alpha*d;
    
    nbeta = nbeta./sum(nbeta);

    if max(nbeta - beta) < 1e-6 || t>maxiter
        flag = 0;
    end

    t = t+1;
    beta = nbeta;
    sbeta{t} = nbeta;
    obj(t) = beta'*M*beta;
    

    
end 

end

