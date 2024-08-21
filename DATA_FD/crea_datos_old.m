%
% ----------------
% MATLAB: data.m
% ----------------
%
% Generates data for the adjoint problem
%  
%

clear variables

clc

tic

g    = 0.9;
c    = 1.;

%
% It used to work for: Lx=Ly=10., nx=ny=51 dt=0.3 tf=80*0.3
%

Lx   = 5.; 
Ly   = 5.; 
nx   = 51;
ny   = 51;
nit  = 250;
dt   = 0.01; 
tf   = nit*dt;
t    = linspace(dt,tf,nit);
dx   = Lx / nx;
dy   = Ly / ny;

x=zeros(1,nx);
for i=1:nx
    x(i)=-(Lx-dx)/2.0+(i-1)*dx;
end

x2=zeros(1,nx+1);
for i=1:nx+1
    x2(i)=-(Lx)/2.0+(i-1)*dx;
end

y=zeros(1,ny);
for i=1:ny
    y(i)=-(Ly-dy)/2.0+(i-1)*dy;
end

y2=zeros(1,ny+1);
for i=1:ny+1
    y2(i)=-(Ly)/2.0+(i-1)*dy;
end

% Breast: mu_a = 0.037–0.110	mu_s*(1-g)= 11.4–13.5
% Tumor:  mu_a = 0.070–0.10	    mu_s*(1-g)= 14.7–17.3

mu_s = 0.1*ones(nx,ny);     % We give a starting values 
mu_a = 0.000 * ones(nx,ny);   % like the "known" background

% for i=1:nx
%     for j=1:ny
%         if sqrt((x(i))^2+(y(j)-1.0)^2) <= 0.5
%         %if sqrt((x(i)-2.5)^2+(y(j)-2.5)^2) <= 0.3
%             mu_s(i,j) = 0.01;
%         end
%     end 
% end
% 
% for i=1:nx
%     for j=1:ny
%         if sqrt((x(i))^2+(y(j)-1.)^2) <= 0.5
%             mu_a(i,j) = 0.0001;
%         end
%     end 
% end

D0 = ones(nx,ny); %0.2832*ones(nx,ny); % 1./(3*((1-g)*mu_s+mu_a));

for i=1:nx
    for j=1:ny
        %if sqrt((x(i)-0.75)^2+(y(j)-0.75)^2) <= 0.5
        %if sqrt((x(i)-2.5)^2+(y(j)-2.5)^2) <= 0.3
            %D0(i,j) = 10;% 0.01821; %
            D0(j,i) = D0(j,i) + 5.*exp(-(x(i)+0.5)^2 -(y(j)-1.5)^2) + 5.*exp(-(x(i)-1)^2 -(y(j)+1.25)^2);
            %D0(i,j) = D0(i,j) + 5.*exp(-0.5*(x(i))^2 -0.5*(y(j))^2);
        %end
    end 
end

save D0 D0
                         
parameters(1)  = mu_a(1,1) ;
parameters(2)  = mu_s(1,1) ;
parameters(3)  =  g   ;
parameters(4)  =  c   ;
parameters(5)  =  nx  ;
parameters(6)  =  ny  ;
parameters(7)  =  Lx  ;
parameters(8)  =  Ly  ;
parameters(9)  =  dt  ;
parameters(10) =  tf  ;
parameters(11) =  nit ;

save parameters parameters

dsx = Lx/(nx-1);
dsy = Ly/(ny-1);

% sourcex = dsx*[6:5:46,nx*ones(1,9),46:-5:6,ones(1,9)]-Lx/2.-dsx;
% sourcey = dsy*[ones(1,9),6:5:46,ny*ones(1,9),46:-5:6]-Ly/2.-dsy;
% [~,ns] = size(sourcex);

%xsource = [6:5:46,nx*ones(1,9),46:-5:6,ones(1,9)];
%ysource = [ones(1,9),6:5:46,ny*ones(1,9),46:-5:6];
xsource = [-2.5*ones([1,8]), -2.5+5*(1:8)/9, 2.5*ones([1,8]), -2.5+5*(1:8)/9];
ysource = [-2.5+5*(1:8)/9,  2.5*ones([1,8]), -2.5+5*(1:8)/9, -2.5*ones([1,8])];
[~,ns] = size(xsource);

% Data characteristics

xdata = [2:nx-1,nx*ones(1,ny-2),nx-1:-1:2,ones(1,ny-2)]; 
ydata = [ones(1,nx-2),2:ny-1,ny*ones(1,nx-2),ny-1:-1:2]; 

[md,nd] = size(xdata);

% Generating data

direct = 1;

for is = 1:ns

    disp(int2str(is))
    pause(0.05)
    
    % Point source

    sol=difusion_test(direct,parameters,[],xdata,...
       ydata,nd,nit,mu_a,D0,is,xsource,ysource);

%     sol=difusion_old(direct,parameters,[],xdata,...
%        ydata,nd,nit,mu_a,D,is,sourcex,sourcey);

    file=['soldata_point_s_' int2str(is)];
    save(file,'sol')
    
    clear sol
        
end  