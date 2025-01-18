clc; clear all; close all; beep off;


%{
 
dS/S = mu*dt+sigma*dW

S = price of a stock at time t
mu > 0, drift
sigma - volatility

solution: S(t) = s0*exp[sigma*W(t)+(mu-sigma^2/2)*t]

%}

mu = 1;
sigma = 1;

u0 = 1; % initial condition
T = 5; % end time
N = 10*5; % number of time intervals.

f = @(x)mu;
g = @(x)sigma;

% [t,u] = euler_maruyama(u0,T,N,f,g);

% plot(t,u,'-k','LineWidth',2);
% h = xlabel('t');
% set(h,'FontSize',18);
% h = ylabel('S(t)');
% set(h,'FontSize',18);
% h = gca;
% set(h,'FontSize',18);


iterations = 10000;
samples = zeros(1,iterations);
for j = 1:iterations
    [t,u] = euler_maruyama(u0,T,N,f,g);
    samples(j) = u(end);
end

h = histfit(samples)



