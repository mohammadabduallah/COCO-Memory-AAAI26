function results = coco_memory_no_prediction( ...
        fML, gradfML, gML, gradgML, ...
        fMEM, gMEM, ...
        m, T, R, bench, benchPer)



    d  = length(gradfML{1}(zeros(1)));   % dimension
    x  = ones(T+1, d);
    V  = zeros(T, 1);


    PhiPrime = @(Vh, lambda) 2*lambda*Vh;       % COCO-M2
    rng(0); for t = 0:m, x(t+1,:) = proj_ball(randn(1,d), R); end

    % statistics
    cCostAlg  = 0;  cViol = 0;
    avgRegret = NaN(T-m,1);
    ccv       = NaN(T-m,1);
    gradHist  = 0;

    % helper to retrieve benchmark without recomputing
    hasBench = (exist('bench','var') && isstruct(bench) && ...
                isfield(bench,'fOpt') && isfield(bench,'xStar') && ...
                numel(bench.fOpt) >= T);

    for t = m:(T-1)
        lambda = 1/sqrt(t);
        xt     = x(t+1,:);
        
        % ----- online gradient step -----
        gt    = gML{t}(xt);   gp = max(0, gt);
        dgp   = (gt > 0) * gradgML{t}(xt);
        Vhat  = V(t) + gp;    V(t+1) = Vhat;

        Lgrad = gradfML{t}(xt) + PhiPrime(Vhat, lambda) * dgp;
        gradHist = gradHist + norm(Lgrad)^2;
        eta   = sqrt(2)*R/(2*sqrt(gradHist));
        x(t+2,:) = proj_ball(xt - eta*Lgrad, R);

        % ----- true memory cost / violation (sliding window) -----

        win  = (t-m+1):(t+1);
        cost = fMEM{t}(x(win+1,:));
        viol = max(0, gMEM{t}(x(win+1,:)));
        
        cCostAlg = cCostAlg + cost;
        cViol    = cViol    + viol;

        % ----- checkpoints: use precomputed benchmark if available -----
        if mod(t, benchPer) == 0

            fOpt  = bench.fOpt(t);

            k            = t - m + 1;  % store at aligned index
            avgRegret(k) = (cCostAlg - fOpt) / t;
            ccv(k)       = cViol / t;
       
        end
    end

    % assemble results
    results = struct('x', x, 'V', V, ...
                     'avgRegret', avgRegret, 'ccv', ccv, ...
                     'benchPer', benchPer);
end

function y = proj_ball(x, R)
    if R < 0, error('R must be nonnegative'); end
    nrm = norm(x);
    if nrm <= R, y = x; else, y = (R/nrm)*x; end
end

































