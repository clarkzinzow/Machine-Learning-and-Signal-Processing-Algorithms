function [V, W, inform] = backPropagation(X, V, W, k, pgparams)
%  Backpropagation algorithm.
%
%  Input:
%    y        - Random projections of the signal w:  y = X'w + nu.
%    X        - 
%    V        - Initial guess for weight V.
%    W        - Initial guess for weight W.
%    k        - Scaling constant for step size: k < 1.
%    pgparams - Parameter structure indicating convergence threholds:
%               * pgparams.maxit = maximum number of iterations.
%               * pgparams.stepthresh = ||w^k - w^k-1|| convergence threshold.

%  Initialize useful parameters, thresholds, and counters.
thresh = pgparams.stepthresh;  % Threshold for successive w-distances.
p = size(X,1)-1;  % Number of features.
n = size(X,2);  % Number of training samples.
delta_W = 1;  % Will hold (2-norm) distances between w_p and w.
delta_V = 1;
maxit = pgparams.maxit;  % Maximum number of iterations that we allow.
t = 1;  % Step number.

% %  Set parameters for dynamically resizing array containing w vectors.
% BLOCK_SIZE = n;  % Array block size.
% w_val_size = BLOCK_SIZE;  % Initial number of w slots.
% w_values = zeros(p, w_val_size);  % Preallocation.
% w_values(:,t) = w;  % Add initial w guess to w vector array.

%  Update w ntil the w-distance threshold is met or we exceed the maximum number
%  of allowed iterations.
% g = inf;
% g_matrix = zeros(p,n);
while ((delta_W > thresh) || (delta_V > thresh)) && (t < maxit + 1)
    V_p = V;  % Set V to previous V value, from last iteration.
    W_p = W;  % Set W to previous W value, from last iteration.
    
%     pg = g;
    i_t = randi(n);  % Choose i_t uniformly at random from {1,...,n}.
    
    %  Calculate h and x values.
    h_t = W'*X(:,i_t);
    x_hat_t = V'*h_t;
    
    %  Choose step size.
    alpha_t = k / sqrt(t);
    
    %  Calculate delta and gamma values.
    delta_t = x_hat_t - X(2:p+1,i_t);
    gamma_t = V*delta_t;  % Assuming that q = p.
    
    %  Optimize the weights for each layer, starting with the deepest layer.
    V = V_p - alpha_t*h_t*delta_t';
    W = W_p - alpha_t*X(:,i_t)*gamma_t';
    
    %  Update convergence checks.
    delta_W = norm(W - W_p)^2;  % Get new successive w-distance.
    delta_V = norm(V - V_p)^2;  % Get new successive v-distance.
    
    %  Increment step number.
    t = t + 1;
    
%     %  Add new w vector to w_values array.
%     w_values(:,t) = w;
%     
%     %  If less than 10% of w_values capacity left, resize.
%     if t + (BLOCK_SIZE/10) > w_val_size
%         w_val_size = w_val_size + BLOCK_SIZE;
%         w_values(:, t+1:w_val_size) = 0;
%     end
end
% w_values(:, t:end) = [];  % Get rid of remaining w_values slots.
% w_val_cols = num2cell(w_values,1);  % Create cell array of w vectors.
% %  Get index (T) of minimum function value.
% [~,T] = min(cellfun(@(u) max(0, 1-(u'*X)*y), w_val_cols));
% w = w_values(:,T);  % Set w^(T) to be the w vector that we return.

%  If we met the threshold, then we obtained the minimum we were searching for.
if (delta_W <= thresh) && (delta_V <= thresh)
    inform.status = 1;  % Set the status of the algorithm to 1, or "success".
else
    inform.status = 0;  % Otherwise, set to 0, or "failed".
end
inform.iter = t-1;  % Store number of iterations endured.
% inform.w_values = w_values;
inform.delta_W = delta_W;
inform.delta_V = delta_V;

% We return w and the inform structure.

end