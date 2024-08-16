%
% ----------------
% MATLAB: difusion_test.m
% ----------------
%
% Compute the solution of the light diffusion equation in two
% dimensions using finite differences. 
%


function sol = difusion_test(direct,parameters,sourced,xdata,ydata,nd,nit,mu_a,D,is,xsource,ysource)


% set the optical properties of the medium

%tic

% g    = parameters(3);
c    = parameters(4);

% set up the numerical grids

nx = parameters(5);
ny = parameters(6);

Lx = parameters(7); 
dx = Lx / (nx - 1);

x = zeros(nx,1);
for i = 1:nx
    x(i) = -Lx / 2.0 + (i - 1) * dx;
end

x2 = zeros(nx+1,1);
for i = 1:nx+1
    x2(i) = -(Lx - dx) / 2.0 + (i - 1) * dx - dx;
end

Ly = parameters(8); 
dy = Ly / (ny - 1);

y = zeros(ny,1);
for i = 1:ny
    y(i) = -Ly / 2.0 + (i-1) * dy;
end

y2 = zeros(ny+1,1);
for i = 1:ny+1
    y2(i) = -(Ly - dy) / 2.0 + (i - 1) * dy - dy;
end

% Dx2 = zeros(nx+1,ny);
% Dy2 = zeros(nx,ny+1);
D_av = sum(sum(D))/(nx*ny);
D_av = 1.
Dx2  = interp2(y,x,D,y,x2','linear',D_av); % the ...,Dav) is for the points outside the domain (extrapolation)
Dy2  = interp2(y,x,D,y2',x,'linear',D_av);

% compute the time step

dt  = parameters(9);
tf  = parameters(10);

% Computation of the differenciation matrix

alpha = dt / dx^2;
beta  = dt / dy^2;

MAT=zeros(nx*ny,nx*ny);
for i=2:nx-1
    for j=2:ny-1
        
        k=(j-1)*nx+i;
        
        MAT(k,k) = 1. + dt*mu_a(i,j) +...
            alpha * (Dx2(i,j) + Dx2(i+1,j)) +...
            beta  * (Dy2(i,j) + Dy2(i,j+1));

        MAT(k,k+1)  = -alpha * Dx2(i+1,j);
        MAT(k,k-1)  = -alpha * Dx2(i,j);
        MAT(k,k+nx) = -beta  * Dy2(i,j+1);
        MAT(k,k-nx) = -beta  * Dy2(i,j);
       
    end
end

i=1;
    for j=2:ny-1
        
        k=(j-1)*nx+i;
        
        MAT(k,k) = 1. + dt*mu_a(i,j) +...
            alpha * (Dx2(i,j) + Dx2(i+1,j)) +...
            beta  * (Dy2(i,j) + Dy2(i,j+1))+...
            alpha * dx*Dx2(i,j) / D(i,j);
            

        MAT(k,k+1)  = -alpha * (Dx2(i+1,j)+Dx2(i,j));
        MAT(k,k+nx) = -beta * Dy2(i,j+1);
        MAT(k,k-nx) = -beta * Dy2(i,j); 
        
    end
    

i=nx;
    for j=2:ny-1
        
        k=(j-1)*nx+i;
        
        MAT(k,k) = 1. + dt*mu_a(i,j) +...
            alpha * (Dx2(i,j) + Dx2(i+1,j)) +...
            beta  * (Dy2(i,j) + Dy2(i,j+1)) +...
            alpha * dx*Dx2(i+1,j) / D(i,j);

        MAT(k,k-1)  = -alpha * (Dx2(i,j)+Dx2(i+1,j));
        MAT(k,k+nx) = -beta  * Dy2(i,j+1);
        MAT(k,k-nx) = -beta  * Dy2(i,j);
    end
    

for i=2:nx-1
    j=1;
        
        k=(j-1)*nx+i;
        
        MAT(k,k) = 1. + dt*mu_a(i,j)+...
            alpha * (Dx2(i,j) + Dx2(i+1,j)) +...
            beta  * (Dy2(i,j) + Dy2(i,j+1))+...
            beta  * dy*Dy2(i,j)/D(i,j);

        MAT(k,k+1)  = -alpha * Dx2(i+1,j);
        MAT(k,k-1)  = -alpha * Dx2(i,j);
        MAT(k,k+nx) = -beta* (Dy2(i,j+1)+Dy2(i,j));
       
end    
    
for i=2:nx-1
    j=ny;
        
        k=(j-1)*nx+i;
        
        MAT(k,k) = 1. + dt*mu_a(i,j)+...
            alpha * (Dx2(i,j) + Dx2(i+1,j)) +...
            beta  * (Dy2(i,j) + Dy2(i,j+1)) + ...
            beta  * dy*Dy2(i,j+1) / D(i,j);

        MAT(k,k+1)  = -alpha * Dx2(i+1,j);
        MAT(k,k-1)  = -alpha * Dx2(i,j);
        MAT(k,k-nx) = -beta * (Dy2(i,j)+Dy2(i,j+1));

end    


i=1;
    j=1;
    
        k=(j-1)*nx+i;
        
        MAT(k,k) = 1. + dt*mu_a(i,j) +...
            alpha * (Dx2(i,j) + Dx2(i+1,j)) +...
            beta  * (Dy2(i,j) + Dy2(i,j+1))+...
            alpha * dx *Dx2(i,j) / D(i,j)+...
            beta  * dy*Dy2(i,j) / D(i,j);
        
        MAT(k,k+1)  = -alpha * (Dx2(i+1,j)+Dx2(i,j));
        MAT(k,k+nx) = -beta * (Dy2(i,j+1)+Dy2(i,j));
        
                
i=nx;
    j=1;
    
        k=(j-1)*nx+i;
        
        MAT(k,k) = 1. + dt*mu_a(i,j) +...
            alpha * (Dx2(i,j) + Dx2(i+1,j)) +...
            beta  * (Dy2(i,j) + Dy2(i,j+1)) +...
            alpha * dx *Dx2(i+1,j) / D(i,j)+...
            beta  * dy *Dy2(i,j)   / D(i,j);

        MAT(k,k-1)  = -alpha * (Dx2(i,j)+Dx2(i+1,j));
        MAT(k,k+nx) = -beta * (Dy2(i,j+1)+Dy2(i,j));
        
        
i=nx;
    j=ny;
    
        k=(j-1)*nx+i;
        
        MAT(k,k) = 1. + dt*mu_a(i,j) +...
            alpha * (Dx2(i,j) + Dx2(i+1,j)) +...
            beta  * (Dy2(i,j) + Dy2(i,j+1)) +...
            alpha* dx *Dx2(i+1,j) / D(i,j) + ...
            beta  * dy *Dy2(i,j+1) / D(i,j);

        MAT(k,k-1)  = -alpha * (Dx2(i,j)+Dx2(i+1,j));
        MAT(k,k-nx) = -beta * (Dy2(i,j)+Dy2(i,j+1));
        
i=1;
    j=ny;
    
        k=(j-1)*nx+i;
    
        MAT(k,k) = 1. + dt*mu_a(i,j) +...
            alpha * (Dx2(i,j) + Dx2(i+1,j)) +...
            beta  * (Dy2(i,j) + Dy2(i,j+1))+...
            alpha * dx *Dx2(i,j)   / D(i,j) + ...
            beta  * dy *Dy2(i,j+1) / D(i,j);

        MAT(k,k+1)  = -alpha * (Dx2(i+1,j)+Dx2(i,j));
        MAT(k,k-nx) = -beta * (Dy2(i,j)+Dy2(i,j+1));

MAT=sparse(MAT);

% alternating source bucle

for is=is

    % set the initial-boundary condition

    U0=zeros(nx,ny);

    t   = 0.;
    n   = 0;
    kk  = 1;
    sol = zeros(nx,ny,nit + 1);
    while abs(t)<abs(tf)  %for n = 1 : nit

        % compute the current time

        n = n+1;
        t = n * dt;

        % Calculation of the source term

        if direct == 1
            if n==1   
                source = zeros(nx,ny);    
                for i=1:nx
                    for j=1:ny
                        source(i,j)=10*exp(-5.*((x(i)-xsource(is))^2+(y(j)-ysource(is))^2));  %exp(-10000.*(t-20.*dt)^2)*
                    end 
                end
    
                for i=1:nx
                    for j=1:ny
                        %U0(i,j)=U0(i,j)+c*dt*source(i,j);
                        U0(i,j)=U0(i,j)+source(i,j);
                    end
                end
        
                u = U0(:);
                sol(:,:,kk) = U0';
            end

        elseif direct == -1
 
            source = zeros(nx,ny);    
            for i=1:nd
                    source(xdata(i),ydata(i))=sourced(n,i);
            end
  
            for i=1:nx
                for j=1:ny
                    U0(i,j)=U0(i,j)+c*dt*source(i,j);
                end
            end
            
            u = U0(:); 


        end

        u_old = u;
       
        % compute the next solution step

        u = MAT \ u_old;

        U0 = reshape(u,nx,ny); 
        
        % plot the solution

%         if direct == -1
%             figure(34);
% %             h = pcolor(y,x,U0);
%             h = pcolor(U0);
%             set(h,'edgecolor','none')
%             colorbar;
%             pause(0.05)
%         end

        % save the solution
        
        if mod(n,1)==0 || n==1
            kk = kk+1;
            sol(:,:,kk) = U0';
        end

    end

    %toc 
       
end

