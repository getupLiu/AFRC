clear;
data =load("webkb.mat");
X=data.X;
alpha=0.04;
lambda=1;
gamma=1;
view_num=numel(X);
feature_dims = cellfun(@(x) size(x, 2), X);
number_anchor=ceil(alpha*feature_dims);
[obj,G] = AFRC(X, number_anchor,100, 0.01,lambda,gamma);
for it = 1:view_num
    % Transpose the intermediate matrix G{it}
    G{it} = G{it}';
    
    % Compute the Gram matrix r(KK)
    KK = G{it}' * G{it};
    
    eigvals = eig(KK);            % Compute eigenvalues
    rho = max(abs(eigvals));      % Take the maximum absolute eigenvalue, i.e., spectral radius
    
    % Compute r = 0.9 / spectral radius
    r = 0.9 / rho;
    
    I = eye(size(G{it}, 2));      % Construct identity matrix I, same size as G^T * G
    
    inv_term = pinv(I - r * (G{it}' * G{it}));   % Compute (I - r * G^T * G)^(-1)
    
    M_hat = G{it} * (inv_term - eye(size(G{it}, 2)));  % Compute G * (...) - I, result size: e-dimensional compatibility
    
    e = ones(size(G{it}, 2), 1);
    
    % Compute the score vector
    Score{it} = M_hat * e;    % Final importance score vector
end
