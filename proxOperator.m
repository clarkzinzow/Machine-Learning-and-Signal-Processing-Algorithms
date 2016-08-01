function w = proxOperator(v, gamma)
%  Returns the proximal operator for Lasso regularization.
    w = sign(v).*max(abs(v) - gamma, 0);
end