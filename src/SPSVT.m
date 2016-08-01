function [U, Sigma, V, inform] = ...
    SPSVT(n1, n2, Omega, P_M, lambda, alpha, s_incr, tol, maxiter)
    %  Recovers a low-rank matrix from a partial sampling of its entries, using
    %  methods that take advantage of the fact that the iteration matrix, Y_k,
    %  is sparse since it vanishes outside of Omega.
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
    
    m = length(Omega);  % Number of observed entries.
    [i, j] = ind2sub([n1 n2], Omega);  % Omega subscripts.
    
    %  Sparse orthogonal projection of M onto the span of matrices that vanish
    %  outside of Omega.
    P_M_sparse = sparse(i, j, P_M, n1, n2, m);
    
    %  NOTE: Since P_M_sparse is sparse, we use normest to get an estimate of
    %        the 2-norm of P_M_sparse instead of getting the exact 2-norm of
    %        P_M_sparse via a call to norm.
    norm_P_M_sparse = normest(P_M_sparse);
%     norm_P_M_sparse = norm(full(P_M_sparse));

    %  Get kicking device integer k0 in order to skip unnecessary initial
    %  computations.
    k0 = ceil(lambda / (alpha * norm_P_M_sparse));
    
%   norm_P_M = norm(P_M);  % Norm of P_M.
    norm_P_M = norm(P_M, 'fro');  % Frobenius norm of P_M.
    
    %  NOTE: We apply a kicking device by noting that, for k0 defined above,
    %        if we defined Y_0 = 0 as is standard, then we clearly have that
    %        X_k = 0 and Y_k = k * alpha * P_M for k = 1:k_0.  Therefore, we
    %        can start our iteration with calculating X_k0+1 from
    %        Y_k0 = k0 * alpha * P_M.
    
    %  We need O_idx to update sparse iteration matrix Y.
%     [~, O_idx] = sort(Omega);  % Sort set of observed entries.
    
    %  Initial values.
    %  Initial value for iteration matrix, skipping computaitons of X_1:X_k0 via
    %  the kicking device described in the above NOTE.
    y = k0 * alpha * P_M;
    %  NOTE:  We need not set y = y(O_idx) since scalar multiplication does not
    %  change the nonzero structure of P_M.
    %  Sparse version of intial value for iteration vector, y.
    Y = sparse(i, j, y, n1, n2, m);
    %  Rank of X_k0-1, which is 0.
    r = 0;
    
%     msg_flag = 0;
    %  Shrinkage iteration.
    for k = 1 : maxiter
        %  The number of singular values of Y to be computed in this iteration.
        s = r + 1;
        
        s_flag = 0;  % Boolean flag that indicates a loop-breaking condition.
        s_err = 0;
        
        %  While s_flag is not set, compute the first s singular values of
        %  Y_sparse, set s_flag if the smallest of the s singular values is
        %  less than or equal to our threshold lambda, and increment s by
        %  a specified increment amount, s_incr.
        %  NOTE:  There could be a faster way to obtain the s largest singular
        %         values of Y_sparse than using Matlab's svds function.  This
        %         is an area that could be optimized via our own implementation
        %         of a subroutine, or by using a third-party option.
        while ~s_flag
           [U, Sigma, V] = svds(Y, s);
           if s > min(size(Sigma))
               s = min(size(Sigma));
               if Sigma(s,s) > lambda
                   s_err = 1;
                   break;
               end
           end
           s_flag = (Sigma(s,s) <= lambda) || (s == min(n1, n2));
           s = min(s + s_incr, min(n1,n2));
        end
        
        if s_err == 1
            warning('No singular values less than or equal to lambda found.');
            break;
        end
        
        sigma = diag(Sigma);  % Singular values of Y_sparse.
        
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
%                                   % values greater than the threshold lambda.

        U = U(:,1:r);  % Given new r, reduce U to n1 x r.
        V = V(:,1:r);  % Given enw r, reduce V to n2 x r.
        
        sigma = sigma(1:r) - lambda;  % Reduce sigma to r x 1, minus lambda.
        Sigma = diag(sigma);  % Set Sigma to the r x r matrix with sigma as its
                              % diagonal.
                              
        %  Compute matrix product for large matrices.
        W = U*diag(sigma);
        x = zeros(m, 1);  % Preallocate x.
        for h = 1:m
            x(h) = W(i(h),:) * (V(j(h),:)');
        end
        
%         D = U*diag(sigma)*V';  % Compute application of soft-thresholding operator.
%         x = D(Omega);  % Indexing by Omega, we obtain our iteration matrix X.
        
        %  Check if relative error on the set of sampled entries is below the
        %  provided tolerance, tol.
%       if norm(x - P_M) / norm_P_M < tol
        rel_error = norm(x - P_M, 'fro') / norm_P_M;
        if rel_error < tol
            break;
%         elseif (~msg_flag) && rel_error < 1e-2
%             fprintf('Looks like it is converging...\n\n');
%             msg_flag = 1;
        end
        
        if k == 100000
            fprintf('100000 iteration report - rel_error:  %.4g\n\n', rel_error);
        end
        
        %  If the iteration matrices are diverging from the solution, then
        %  output a divergence message and break.
        if norm(x - P_M) / norm_P_M > 1e5
            disp('Diverging, iteration stopped.');
            break;
        end
        
        %  Update y for this iteration.
        y = y + alpha*(P_M - x);
        %  Update sparse iteration matrix, Y.
        Y = sparse(i, j, y, n1, n2);
        %  NOTE:  Since we do only update nonzero elements in Y_sparse,
        %         we avoid the cumbersome overhead of changing the nonzero
        %         pattern in a sparse matrix.
%         Y = Y(O_idx);
%         for i = 1:m
%             j = O_idx(i);
%             
%             %  We suppress the Matlab Code Analyzer warning since we only update
%             %  nonzero elements in Y_sparse (as ensured by our appealing to the
%             %  sorted Omega linear index vector, O_idx.)
%             Y(i) = y(j); %#ok<SPRIX>
%         end
%         Y = Y(O_idx);
%         [O_idx_i, O_idx_j, ~] = find(Y_sparse);
%         Y_sparse = sparse(O_idx_i, O_idx_j, Y, n1, n2);
    end
    
    %  Set inform structure with algorithm results.
    
    %  Return the number of iterations.
    inform.numiter = k;
    
    %  If convergence criteria was met, set status of algorithm to 1.
%   if norm(X - P_M) / norm_P_M < tol
    if norm(x - P_M, 'fro') / norm_P_M < tol
        inform.status = 1;
    %  Otherwise, set status of algorithm to 0.
    else
        inform.status = 0;
    end
    
    %  Return the rank of the optimal matrix.
    inform.rank = r;
    
end