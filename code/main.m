data_name = 'proteinFold';
data_path = 'D:\Work\datasets\Kmatrix\';
load([data_path, data_name, '_Kmatrix.mat'], 'KH');

numker = size(KH, 3);
K = kcenter(KH);
K = knorm(K);

M = zeros(numker);
for p=1:numker
    for q=1:numker
        M(p,q) = trace(K(:,:,p)*K(:,:,q));
    end
end
M = (M+M')/2;

% ----------------------------------------------------------
% QP problem: quadprog()
H = M;
f = [];
A = [];
b = [];
Aeq = ones(1,numker);
beq = 1;
lb = zeros(numker,1);
ub = ones(numker,1);
[beta_1,fval,exitflag] = quadprog(H,f,A,b,Aeq,beq,lb,ub);


% ----------------------------------------------------------
% QP problem: RGD_1
maxiter = 1e5;
[beta_2, sbeta_2, obj_2] = RGD_demo_1(M, maxiter);

% ----------------------------------------------------------
% QP problem: RGD_2
maxiter = 1e5;
[beta_3, sbeta_3, obj_3] = RGD_demo_2(M, maxiter);

figure(1)
set(gcf,'position',[0,0,1000,400]);
bar([beta_1,beta_2,beta_3]);
ylim([0, 0.5]);
% legend('quadprog()', 'RGD 1', 'RGD 2');
legend('\fontsize{12}{quadprog()}', '\fontsize{12}{RGD 1}', '\fontsize{12}{RGD 2}');
title('\fontsize{16}{Protein Fold}');
xlabel('\fontsize{16}{Index}');
ylabel('\fontsize{16}{\beta}');
axis on;
grid on;


figure(2)
set(gcf,'position',[0,0,1000,500]);
sp1 = subplot(1,2,1);
plot(obj_2,'LineWidth',2);
hold on;
title('\fontsize{16}{RGD 1 on Protein Fold}');
xlabel('\fontsize{16}{Index}');
ylabel('\fontsize{16}{Objective Value}');
axis square;
grid on;
sp2 = subplot(1,2,2);
plot(obj_3,'LineWidth',2);
hold on;
title('\fontsize{16}{RGD 2 on Protein Fold}');
xlabel('\fontsize{16}{Index}');
ylabel('\fontsize{16}{Objective Value}');
axis square;
grid on;



