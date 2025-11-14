function [fML, gradfML, gML, gradgML, fMEM, gMEM, C, D] = ...
         simple_random_coco(d, T, m, sigma, delta, prob, seed, gamma, type)
% SIMPLE_RANDOM_COCO  — bounded‑parameter test‑bed generator
%
% d        : dimension
% T        : horizon
% m        : memory length
% sigma    : scale of the underlying Normal / Uniform draws
% delta    : affine offset in g_t(x)=d_t^T x - delta
% prob     : P[Normal draw]   (else Uniform)         — used only if type==1
% seed     : RNG seed  (default 1)
% gamma    : truncation factor (default 3 ⇒ 99.7 % of N(0,1))
% type     : 1 → "adversarial'' mix  |  2 → pure Uniform (stochastic)
%
% OUTPUT
%   fML{t}, gradfML{t} : memory‑less cost and gradient handles
%   gML{t}, gradgML{t} : memory‑less constraint and gradient handles
%   fMEM{t}, gMEM{t}   : memory‑averaged handles over the window size m+1
%   C{t}, D{t}         : raw coefficient vectors used in cost/constraint

if nargin < 8,  gamma = 3;   end
if nargin < 7,  seed  = 1;   end
rng(seed);

B     = sigma * gamma;                    % |c_t(i)|, |d_t(i)| ≤ B
clip  = @(v) max(-B, min(B, v));          % helper for truncation

% Preallocate
fML = cell(T,1);  gradfML = cell(T,1);
gML = cell(T,1);  gradgML = cell(T,1);
fMEM = cell(T,1); gMEM = cell(T,1);
C = cell(T,1);    D = cell(T,1);

% Generate problem instance
for t = 1:T
    % ----------- draw coefficients --------------------------------
    switch type
        case 1  % adversarial: Normal vs. Uniform mix
            if rand < prob
                c_val = clip(sigma * randn(d,1));
                d_val = clip(sigma * randn(d,1));
            else
                c_val = clip(sigma * (2*rand(d,1) - 1));
                d_val = clip(sigma * (2*rand(d,1) - 1));
            end
        otherwise  % pure Uniform
            c_val = clip(sigma * (2*rand(d,1) - 1));
            d_val = clip(sigma * (2*rand(d,1) - 1));
    end

    % Store coefficients
    C{t} = c_val;
    D{t} = d_val;

    % Create memoryless handles (capturing values)
    fML{t}     = @(x) 0.5 * norm(x(:) - c_val)^2;
    gradfML{t} = @(x) (x(:) - c_val).';
    gML{t}     = @(x) d_val.' * x(:) - delta;
    gradgML{t} = @(x) d_val.';
end

% Memory-aware function handles
for t = 1:T
    fMEM{t} = @(X) mean( arrayfun(@(i) fML{t}(X(i,:)), 1:(m+1) ) );
    gMEM{t} = @(X) mean( arrayfun(@(i) gML{t}(X(i,:)), 1:(m+1) ) );
end
end
