function [U, Sigma, V, inform] = SVT(n1, n2, Omega, P_M, lambda, alpha, ...
    s_incr, sp_thresh, sz_thresh, tol, maxiter)
    %  Recovers a low-rank matrix from a partial sampling of its entries.
    %
    %  Inputs:
    %    n1        - Number of rows in matrix X.
    %    n2        - Number of columns in matrix X.
    %    Omega     - Set of observed entries.
    %    P_M       - Orthogonal projection of M onto the span of matrices that
    %              vanish outside of Omega, in the vector form of M(Omega).
    %    lambda    - Singular value threshold.
    %    alpha     - Step size.
    %    s_incr    - Size of incrememnt for s_k, the number of singular values of
    %                Y^(k-1) to be computed at the kth iteration. 
    %    sp_thresh - Sparsity threshold.
    %    sz_thresh - Size threshold.
    %    tol       - Tolerance for relative error on the set of sampled entries.
    %    maxiter   - Maximum number of iterations as fallback stopping condition.
    %
    %  Outputs:
    %    U       - n1 x r left singular vectors.
    %    Sigma   - r x 1 singular values.
    %    V       - n2 x r right singular vectors.
    %    inform  - Structure containing other results of the algorithm.

    density = nnz(P_M) / (n1 * n2);  % Density of P_M.
    %  If the density of P_M is less than the sparsity threshold and if the
    %  size of M is greater than the size threshold, run the sparse version of
    %  the SVT algorithm.
    if((density < sp_thresh) && (n1*n2 > sz_thresh))
        [U, Sigma, V, inform] = SPSVT(n1, n2, Omega, P_M, lambda, alpha, ...
           s_incr, tol, maxiter);
       
        %  Return whether the sparse or non-sparse SVT algorithm was run.
        inform.alg = 'SPSVT';
   %  Otherwise, run full version of SVT algorithm.
    else
        [U, Sigma, V, inform] = FSVT(n1, n2, Omega, P_M, lambda, alpha, ...
           tol, maxiter);
       
        %  Return whether the sparse or non-sparse SVT algorithm was run.
        inform.alg = 'FSVT';
    end
    inform.density = density;
    
end