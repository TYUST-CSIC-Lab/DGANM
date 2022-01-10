function varargout = lianbangs(Operation,Global,input)
% <problem> <DTLZ>
% Scalable Test Problems for Evolutionary Multi-Objective Optimization
% operator --- EAreal

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    switch Operation
        case 'init'
            Global.M        = 4;
            Global.D        = 5;
            Global.lower   = [200,400,900,0.5,0.1];
            Global.upper    = [300,600,1100,0.9,0.5];
            Global.operator = @EAreal;        
            PopDec = importdata('C:\Users\HPC\Desktop\LY\morematlab\paramater3.txt');
            varargout = {PopDec};
       case 'value'
            PopDec = input;
            [~,D]  = size(PopDec);
            M      = Global.M;
            %% 目标函数计算                    
            
            if count(py.sys.path,'') == 0
                insert(py.sys.path,int32(0),'C:\Users\HPC\Desktop\LY\morematlab\DTLZ');
            end
           [N,D] = size(input);
           dlmwrite('C:\Users\HPC\Desktop\LY\morematlab\N.txt',  N, 'delimiter', ',', 'newline', 'pc');
           dlmwrite('C:\Users\HPC\Desktop\LY\morematlab\paramater3.txt',  PopDec, 'delimiter', ',', 'newline', 'pc');
           data = importdata('C:\Users\HPC\Desktop\LY\morematlab\paramater3.txt');

           dlmwrite('C:\Users\HPC\Desktop\LY\morematlab\paramater3.txt',  data, 'delimiter', ',', 'newline', 'pc');
           result1 = python1('C:\Users\HPC\Desktop\LY\morematlab\paramater3.txt');
%            result1
           loss=importdata('C:\Users\HPC\Desktop\LY\morematlab\loss.txt');
           IS=importdata('C:\Users\HPC\Desktop\LY\morematlab\IS.txt');
           FID=importdata('C:\Users\HPC\Desktop\LY\morematlab\FID.txt');


           for i = 1:N
                Fitness1(i,:) =loss(i,1);
                Fitness2(i,:) = IS(i,1);
                Fitness3(i,:) =IS(i,2);
                Fitness4(i,:) = FID(i,1);

           end
        
             %% 总结 
           F1 = [Fitness1,Fitness2,Fitness3,Fitness4];
           %PopObj = [Fitness1,Fitness2,Fitness3,Fitness4]
           %F2 = [Fitness1,Fitness2,Fitness3,Fitness4];  %原始数据
           %归一化
           [m1,m_index1] = max(Fitness1);
           [n1,n_index1] = min(Fitness1);
           [m2,m_index2] = max(Fitness2);
           [n2,n_index2] = min(Fitness2);
           [m3,m_index3] = max(Fitness3);
           [n3,n_index3] = min(Fitness3);
           [m4,m_index4] = max(Fitness4);
           [n4,n_index4] = min(Fitness4);
           
           for i = 1:N
               Fitness1(i,:) = (Fitness1(i,:)-n1)/(m1-n1);
               Fitness2(i,:) = (Fitness2(i,:)-n2)/(m2-n2);
               Fitness3(i,:) = (Fitness3(i,:)-n3)/(m3-n3);
               Fitness4(i,:) = (Fitness4(i,:)-n4)/(m4-n4);
              
           end          
           PopObj=[Fitness1,Fitness2,Fitness3,Fitness4];  %归一化
%            if mod(Global.gen,10) ==0
%                 PopDec
%            end
           PopCon = [];
            varargout = {input,PopObj,PopCon};
        case 'PF'
            f = UniformPoint(input,Global.M)/2;
            varargout = {f};
    end
end