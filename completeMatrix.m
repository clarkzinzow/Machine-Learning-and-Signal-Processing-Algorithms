function [X_opt, inform] = completeMatrix(n1, n2, Omega, P_M, tol, maxiter)
    %  Reconstructs a low-rank matrix from a partial sampling of its entries.
    %
    %  Inputs:
    %    n1        - Number of rows in matrix X.
    %    n2        - Number of columns in matrix X.
    %    Omega     - Set of observed entries.
    %    P_M       - Orthogonal projection of M onto the span of matrices that
    %                vanish outside of Omega, in the vector form of M(Omega).
    %    tol       - Tolerance for relative error on the set of sampled entries.
    %    maxiter   - Maximum number of iterations as fallback stopping condition.
    %
    %  Outputs:
    %    X_opt   - The recovered low-rank matrix.
    %    inform  - Structure containing other results of the algorithm.
    
    %  Establish non-user provided parameters for SVT algorithm.
    m = length(Omega);  % Number of sampled entries.
    %  TODO:  Implement stepsize-finding functionality for each iteration in
    %         FSVT.m and SPSVT.m.
    alpha = (1.2 * n1 * n2) / m;  % Constant step size.
    %  The following value for lambda ensures that, on average, lambda*||M||_*
    %  is about 10 times larger than 0.5*||M||_F^2, provided that the rank is
    %  bounded away from the dimensions n1 and nn2.
    %  TODO:  Implement cross-validation for lambda.  Does that yield better
    %         results?
    lambda = 5*sqrt(n1*n2);
    %  TODO:  Experiement with other values for s_incr.
    s_incr = 5;  % Found to work well in practice.
    %  TODO:  Determine if this is the best sparsity threshold to use.
    sp_thresh = 0.5;  % Sparsity threshold.
    sz_thresh = 100*100;  % Size threshold.
    
    %  Call the general SVT algorithm with the provided data and the
    %  established parameters.
    [U, Sigma, V, inform] = SVT(n1, n2, Omega, P_M, lambda, alpha, s_incr, ...
        sp_thresh, sz_thresh, tol, maxiter);    
    %  Return the recovered matrix, X.
    X_opt = U*Sigma*V';

end