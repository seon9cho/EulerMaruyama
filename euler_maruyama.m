function [t,u] = euler_maruyama(u0,T,N,f,g)

%{
    u0 - initial condition
    T - end time
    N - number of time intervals
    f - function handle
    g - function handle
%}

% du = f(u)dt + g(u)dW(t)

delt = T/N; % time step
dim = length(u0); % dimension of the ODE
u = zeros(dim,N+1); % solution variable
t = (0:delt:T).'; % time mesh
scale = sqrt(delt); % scalar to convert N(0,1) to N(0,delt)
u(:,1) = u0; % initial condition

for n = 1:N % time advance
   dW = scale*randn(dim,1); % change in Brownian motion
   u(:,n+1) = u(:,n)+delt*f(u(:,n))+g(u(:,n))*dW; % solution at next time
end








