function [w, inform] = stochasticGradientDescent(y, X, w, pgparams)
%  Stochastic gradient descent algorithm learning p-dimensional signal w
%  measured as y = X'w + nu, where the columns of X are uniformly distributed
%  p-dimensional vectors in [-1,1].
%
%  Input:
%    y        - Random projections of the s-sparse signal w:  y = X'w.
%    X        - 
%    w        - Initial arbitrary guess for signal w.
%    pgparams - Parameter structure indicating convergence threholds:
%               * pgparams.maxit = maximum number of iterations.
%               * pgparams.stepthresh = ||w^k - w^k-1|| convergence threshold.

%  Initialize useful parameters, thresholds, and counters.
thresh = pgparams.stepthresh;  % Threshold for successive w-distances.
n = length(y);  % Number of training samples.
p = size(X,1);  % Number of features.
delta = 1;  % Will hold (2-norm) distances between w_p and w.
maxit = pgparams.maxit;  % Maximum number of iterations that we allow.
t = 1;  % Step number.

%  Set parameters for dynamically resizing array containing w vectors.
BLOCK_SIZE = n;  % Array block size.
w_val_size = BLOCK_SIZE;  % Initial number of w slots.
w_values = zeros(p, w_val_size);  % Preallocation.
w_values(:,t) = w;  % Add initial w guess to w vector array.

%  Update w ntil the w-distance threshold is met or we exceed the maximum number
%  of allowed iterations.
% g = inf;
% g_matrix = zeros(p,n);
while (delta > thresh) && (t < maxit + 1)
    w_p = w;  % Set w to previous w value, from last iteration.
%     pg = g;
    i_t = randi(n);  % Choose i_t uniformly at random from {1,...,n}.
    
    % Calculate the gradient at the previous w value.
    g_t = -y(i_t)*X(:,i_t)*(y(i_t)*(w_p'*X(:,i_t)) < 1);
%     g_t = 2*(X(:,i_t)*X(:,i_t)'*w_p - X(:,i_t)*y(i_t));
    
%     for i = 1:n
%         g_matrix(:,i) = -y(i)*X(:,i)*(y(i)*(w_p'*X(:,i)) < 1);
%     end
%     g = sum(g_matrix,2);
    
%     fprintf('  gradient: '); fprintf('%8.4g \n', g_t);
    
    %  Update w.
    w = w_p - (1/sqrt(t))*g_t;  % Get proximal operator.
    
%     fprintf('  w: '); fprintf('%8.4g \n', w);
    
    %  Update convergence checks.
    delta = norm(w - w_p)^2;  % Get new successive w-distance.
    
    %  Increment step number.
    t = t + 1;
    
    %  Add new w vector to w_values array.
    w_values(:,t) = w;
    
    %  If less than 10% of w_values capacity left, resize.
    if t + (BLOCK_SIZE/10) > w_val_size
        w_val_size = w_val_size + BLOCK_SIZE;
        w_values(:, t+1:w_val_size) = 0;
    end
end
w_values(:, t:end) = [];  % Get rid of remaining w_values slots.
w_val_cols = num2cell(w_values,1);  % Create cell array of w vectors.
%  Get index (T) of minimum function value.
[~,T] = min(cellfun(@(u) max(0, 1-(u'*X)*y), w_val_cols));
w = w_values(:,T);  % Set w^(T) to be the w vector that we return.

%  If we met the threshold, then we obtained the minimum we were searching for.
if delta <= thresh
    inform.status = 1;  % Set the status of the algorithm to 1, or "success".
else
    inform.status = 0;  % Otherwise, set to 0, or "failed".
end
inform.iter = t-1;  % Store number of iterations endured.
inform.w_values = w_values;
inform.delta = delta;

% We return w and the inform structure.

end