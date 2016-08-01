function [w, inform] = proximalGradientLasso(y, X, w, beta, lambda, pgparams)
%  Proximal gradient algorithm recovering s-sparse p-dimensional signal w
%  measured as y = X'w, where the columns of X are white Gaussian noise.  In
%  other words, we minimize
%                 ||y - X'w||^2 + lambda ||w||_1
%  over w.
%
%  Input:
%    y        - Random projections of the s-sparse signal w:  y = X'w.
%    X        - White Gaussian noise matrix.
%    w        - Initial arbitrary guess for the s-sparse signal, w.
%    beta     - Fixed shrinking parameter for backtracking line search.
%    lambda   - Regularization parameter.
%    pgparams - Parameter structure indicating convergence threholds:
%               * pgparams.maxit = maximum number of iterations.
%               * pgparams.stepthresh = ||w^k - w^k-1|| convergence threshold.

%  Initialize useful parameters, thresholds, and counters.
thresh = pgparams.stepthresh;  % Threshold for successive w-distances.
delta = 1;  % Will hold (2-norm) distances between w_p and w.
maxit = pgparams.maxit;  % Maximum number of iterations that we allow.
num_iter = 0;  % Number of iterations.

%  Precomputations for the sake of performance.
pre_X_prod = 2*(X*X');
pre_Xy_prod = 2*(X*y);

% grads = zeros(maxit,1);  % debugging
% w_values = zeros(maxit,1);  % debugging
% t_values = zeros(maxit,1);  % debugging
% deltas = zeros(100,1);  % Vector to hold delta values.

%  Update w ntil the w-distance threshold is met or we exceed the maximum number
%  of allowed iterations.
while delta > thresh && num_iter < maxit
    w_p = w;  % Set w to previous w value, from laster iteration.
    g = pre_X_prod*w_p - pre_Xy_prod;  % Calculate the gradient at the previous w value.
    t = 1;  % Initially set t parameter to 1.
    
    %  while backtracking line search condition is satisfied, continue to shrink
    %  t.
    while norm(y - X'*proxOperator(w_p - t*g, lambda*t))^2 > ...
            norm(y - X'*w_p)^2 - g'*(w_p - proxOperator(w_p - t*g, lambda*t)) + ...
            0.5*t*norm((w_p - proxOperator(w_p - t*g, lambda*t))/t)^2
        t = beta*t;  % Shrink t by beta*t, where 0 < beta < 1.
    end
    
    %  Update w.
    w = proxOperator(w_p - t*g, lambda*t);  % Get proximal operator.
    
    %  Update convergence checks.
    delta = norm(w - w_p)^2;  % Get new successive w-distance.
    num_iter = num_iter + 1;  % Increment number of iterations.
%     w_values(num_iter) = norm(w);
%     grads(num_iter) = norm(g);
%     deltas(num_iter) = delta;  % Store delta value.
%     t_values(num_iter) = t;
end
%  If we met the threshold, then we obtained the minimum we were searching for.
if delta <= thresh
    inform.status = 1;  % Set the status of the algorithm to 1, or "success".
else
    inform.status = 0;  % Otherwise, set to 0, or "failed".
end
inform.iter = num_iter;  % Store number of iterations endured.
% inform.deltas = deltas;  % Store vector of delta values.
% inform.w_values = w_values;
% inform.grads = grads;
% inform.t_values = t_values;

% We return w and the inform structure.

end