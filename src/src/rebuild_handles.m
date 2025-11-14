function [fML, gradfML, gML, gradgML, fMEM, gMEM] = ...
    rebuild_handles(C, D, T, m, delta)
% REBUILD_HANDLES — Reconstruct COCO function handles from raw data
%
% INPUTS:
%   C{t}      — cell array of cost centers c_t ∈ ℝᵈ
%   D{t}      — cell array of constraint directions d_t ∈ ℝᵈ
%   T         — number of rounds
%   m         — memory window size
%   delta     — constraint offset in g_t(x) = d_tᵀ x - delta
%
% OUTPUTS:
%   fML{t}, gradfML{t} — memoryless cost and gradient
%   gML{t}, gradgML{t} — memoryless constraint and gradient
%   fMEM{t}, gMEM{t}   — memory-averaged cost and constraint

% Initialize outputs
fML     = cell(T,1); gradfML = cell(T,1);
gML     = cell(T,1); gradgML = cell(T,1);
fMEM    = cell(T,1); gMEM    = cell(T,1);

% Rebuild memoryless handles
for t = 1:T
    c_val = C{t};
    d_val = D{t};

    fML{t}     = @(x) 0.5 * norm(x(:) - c_val)^2;
    gradfML{t} = @(x) (x(:) - c_val).';
    gML{t}     = @(x) d_val.' * x(:) - delta;
    gradgML{t} = @(x) d_val.';
end

% Rebuild memory-aware handles
for t = 1:T
    fMEM{t} = @(X) mean(arrayfun(@(i) fML{t}(X(i,:)), 1:(m+1)));
    gMEM{t} = @(X) mean(arrayfun(@(i) gML{t}(X(i,:)), 1:(m+1)));
end
end

