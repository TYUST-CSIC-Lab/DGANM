function NSGAIII(Global)
% <algorithm> <H-N>
% An Evolutionary Many-Objective Optimization Algorithm Using
% Reference-point Based Non-dominated Sorting Approach, Part I: Solving
% Problems with Box Constraints

%--------------------------------------------------------------------------
% The copyright of the PlatEMO belongs to the BIMK Group. You are free to
% use the PlatEMO for research purposes. All publications which use this
% platform or any code in the platform should acknowledge the use of
% "PlatEMO" and reference "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu
% Jin, PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective
% Optimization, 2016".
%--------------------------------------------------------------------------

% Copyright (c) 2016-2017 BIMK Group

    %% Generate the reference points and random population
    [Z,Global.N] = UniformPoint(Global.N,Global.M);
    Population   = Global.Initialization();
    Zmin         = min(Population(all(Population.cons<=0,2)).objs,[],1);

    %% Optimization
    while Global.NotTermination(Population)
%         delete('paramater.csv');
        MatingPool = TournamentSelection(2,Global.N,sum(max(0,Population.cons),2));%锦标赛算法
        Offspring  = Global.Variation(Population(MatingPool));%突变
        Zmin       = min([Zmin;Offspring(all(Offspring.cons<=0,2)).objs],[],1);
        Population = EnvironmentalSelection([Population,Offspring],Global.N,Z,Zmin);
%         for i=1:Global.N
%             Population(1,i) = round(Offspring(1,i).dec);
%         end  
%         PopDec =reshape([Population.dec],[4,10]);  
%         PopDec = PopDec.';
%         dlmwrite('C:\Users\HPC\Desktop\LY\morematlab\paramaters.csv',  PopDec.', 'delimiter', ' ', 'newline', 'pc');
%         paramaters=table([PopDec(:,1),PopDec(:,2),PopDec(:,3),PopDec(:,4)]);
%         writetable(paramaters, 'C:\Users\HPC\Desktop\LY\morematlab\paramaters.csv');
%         result1 = python1('C:\Users\HPC\Desktop\LY\morematlab\paramaters.csv');
%         result1 = importdata('C:\Users\HPC\Desktop\LY\morematlab\result1.csv');
%         result = importdata('C:\Users\HPC\Desktop\LY\morematlab\result.csv');
%         data = importdata('C:\Users\HPC\Desktop\LY\morematlab\paramater.csv');
%         data1 = readtable('C:\Users\HPC\Desktop\LY\morematlab\paramaters.csv');
%         delete('paramaters.csv');
%         delete('result.csv');   
%         delete('result1.csv');
%         data1 = table2array(data1);
%         fit1 = [result1(:,1);result(:,1)].' ;
%         fit2 = [result1(:,2);result(:,2)].';
%         fit3 = [result1(:,3);result(:,3)].';
%         fit4 = [result1(:,4);result(:,4)].';
%         Fit1 = [data1;data];
%         [a,b] = sort(fit1,'descend');%降序
%         [c,d] = sort(fit2);
%         [A,B] = sort(fit3,'descend');
%         [C,D] = sort(fit4,'descend');
%         f=[];
%         for ii = 1:40
%             e1 = find(c==fit2(b(ii)));
%             e2 = find(A==fit3(b(ii)));
%             e3 = find(C==fit4(b(ii)));
%             f(end+1) = [(e1+e2+e3+ii)/4];
%         end
%         [g,h] = sort(f);
%         fit5 = [];
%         fit6 = [];
%         fit7 = [];
%         fit8 = [];
%         fit9 = [];
%         for iii = 1:20
%             fit5(end+1) = fit1(h(iii));
%             fit6(end+1) = fit2(h(iii));
%             fit7(end+1) = fit3(h(iii));
%             fit8(end+1) = fit4(h(iii));
%             fit9 = [fit9;Fit1(h(iii),:)];
%         end
%         result = [fit5.' fit6.' fit7.' fit8.'];
%         dlmwrite('E:\matlab_2016b\多目标\多目标\result.csv',  result, 'delimiter', ' ', 'newline', 'pc');
%         dlmwrite('E:\matlab_2016b\多目标\多目标\paramater.csv',  fit9, 'delimiter', ' ', 'newline', 'pc');
% 
    end
end