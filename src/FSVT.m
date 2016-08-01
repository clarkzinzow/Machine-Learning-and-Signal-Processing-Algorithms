function [U, Sigma, V, inform] = ...
    FSVT(n1, n2, Omega, P_M, lambda, alpha, tol, maxiter)
%     FSVT(n1, n2, Omega, P_M, lambda, alpha0, beta, tol, maxiter)
    %  Recovers a low-rank matrix from a partial sampling of its entries.
    %
    %  Inputs:
    %    n1      - Number of rows in matrix X.
    %    n2      - Number of columns in matrix X.
    %    Omega   - Set of observed entries.
    %    P_M     - Orthogonal projection of M onto the span of matrices that
    %              vanish outside of Omega, in the vector form of M(Omega).
    %    lambda  - Singular value threshold.
    %    alpha   - Step size.
    %    s_incr  - Size of incrememnt for s_k, the number of singular values of
    %              Y^(k-1) to be computed at the kth iteration. 
    %    tol     - Tolerance for relative error on the set of sampled entries.
    %    maxiter - Maximum number of iterations as fallback stopping condition.
    %
    %  Outputs:
    %    U       - n1 x r left singular vectors.
    %    Sigma   - r x 1 singular values.
    %    V       - n2 x r right singular vectors.
    %    inform  - Structure containing other results of the algorithm.

%     P_M_full = zeros(n1,n2);
%     P_M_full(Omega) = P_M;
%     [i, j] = ind2sub([n1 n2], Omega);  % Omega subscripts.
%     subs = horzcat(i', j');  % Concatenate subscripts into r x 2 matrix.
%     %  Convert orthogonal projection vector P_M into n1 x n2 matrix form.
%     P_M_full = accumarray(subs, P_M, [n1 n2]);
    
    
%     norm_P_M = norm(P_M_full, 'fro');  % Frobenius norm of P_M.
    norm_P_M = norm(P_M, 'fro');  % Frobenius norm of P_M.
%     norm_P_M = norm(P_M_full);  % Norm of P_M.

    %  Get kicking device integer k0 in order to skip unnecessary initial
    %  computations.
    k0 = ceil(lambda / (alpha * norm_P_M));
    
    %  NOTE: We apply a kicking device by noting that, for k0 defined above,
    %        if we defined Y_0 = 0 as is standard, then we clearly have that
    %        X_k = 0 and Y_k = k * alpha * P_M for k = 1:k_0.  Therefore, we
    %        can start our iteration with calculating X_k0+1 from
    %        Y_k0 = k0 * alpha * P_M.
    
    %  Initial values.
    %  Initial value for iteration matrix, skipping computaitons of X_1:X_k0 via
    %  the kicking device described in the above NOTE.
    y = k0 * alpha * P_M;
    Y = zeros(n1, n2);
    Y(Omega) = y;
%     Y = k0 * alpha * P_M_full;
    
    %  Rank of X_k0-1, which is 0.
    r = 0;
    
    %  Shrinkage iteration.
    for k = 1 : maxiter
        %  NOTE:  Incrementally finding more and more singular values is only
        %         desirable for sparse matrices, where full singular value
        %         decomposition is usually unnecessary.

        %  Get singular values via singular value decomposition.
        [U, Sigma, V] = svd(Y, 'econ'); 
        
        sigma = diag(Sigma);  % Singular values of Y.
        
        %  NOTE:  r can be found by finding the largest index j in the sigma
        %         vector such that sigma(j) > lambda.  This is due to the fact
        %         that we are guaranteed, by the breaking condition of the
        %         above while loop, that there is at least one j such that
        %         sigma(j) > lambda, and we know that the singular values in
        %         sigma are in descending order, so finding the greatest j such
        %         that sigma(j) > lambda tells us that sigma(i) > lambda for all
        %         i < j and sigma(j) <= lambda for all i > j, hence j is the
        %         number of singular values greater than lambda, hence r = j.
        r = find(sigma > lambda, 1, 'last' );  % The rank of X; i.e., the number
                                               % of singular values greater than
                                               % the threshold lambda.
%         r = sum(sigma > lambda);  % The rank of X; i.e., the number of singular
%                                   values greater than the threshold lambda.

        U = U(:,1:r);  % Given new r, reduce U to n1 x r.
        V = V(:,1:r);  % Given enw r, reduce V to n2 x r.
        
        sigma = sigma(1:r) - lambda;  % Reduce sigma to r x 1, minus lambda.
        Sigma = diag(sigma);  % Set Sigma to the r x r matrix with sigma as its
                              % diagonal.
        
        D = U*diag(sigma)*V';  % Compute application of soft-thresholding operator.
        x = D(Omega);
%         X = zeros(n1,n2);
%         X(Omega) = D(Omega);  % Indexing by Omega, we obtain our iteration matrix X.
        
        %  Check if relative error on the set of sampled entries is below the
        %  provided tolerance, tol.
%         if norm(X - P_M_full) / norm_P_M < tol
        if norm(x - P_M, 'fro') / norm_P_M < tol
%         if norm(X - P_M_full, 'fro') / norm_P_M < tol
            break;
        end
        
        %  If the iteration matrices are diverging from the solution, then
        %  output a divergence message and break.
        if norm(x - P_M) / norm_P_M > 1e5
            disp('Diverging, iteration stopped.');
            break;
        end
        
        %  TODO:  Implement dynamic stepsize selection here.
        
%         alpha = alpha0;
%         while norm(X - proxOp(Y + alpha*(P_M_full - X), lambda), 'fro')^2 <= norm(Y - X, 'fro')^2 + ...
%               (Y - X)*(X - proxOp(Y + alpha*(P_M_full - X), lambda)) + ...
%               lambda/2 * norm(1/lambda * (Y - proxOp(Y + alpha*(P_M_full - X), lambda)), 'fro')
%            alpha = beta*alpha; 
%         end
        
        %  Update y and the iteration matrix, Y.
        y = y + alpha * (P_M - x);
        Y(Omega) = y;
%         %  Update Y for this iteration.
%         Y = Y + alpha*(P_M_full - X);
    end
    
    %  Set inform structure with algorithm results.
    
    %  Return the number of iterations.
    inform.numiter = k;
    
    %  If convergence criteria was met, set status of algorithm to 1.
%     if norm(X - P_M_full) / norm_P_M < tol
    if norm(x - P_M, 'fro') / norm_P_M < tol
%     if norm(X - P_M_full, 'fro') / norm_P_M < tol
        inform.status = 1;
    %  Otherwise, set status of algorithm to 0.
    else
        inform.status = 0;
    end
    
    %  Return the rank of the optimal matrix.
    inform.rank = r;
    
end