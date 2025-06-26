function obj = compute_objective(X, H, W, G, H_star, alpha, lambda, gamma)
    V = length(X);
    obj = 0;  % 初始化目标值

    for v = 1:V
        Hv = H{v};      % m_v × n
        Wv = W{v};      % m × m_v
        Gv = G{v};      % m_v × d_v
        Xv = X{v};      % d_v × n
        av = alpha(v);  % 标量

        % 项1：alpha_v^2 * || Wv*Hv - H_star ||_F^2
        term1 = av^2 * norm(Wv * Hv - H_star, 'fro')^2;

        % 项2：lambda * || Xv - Hv' * Gv ||_F^2
        term2 = lambda * norm(Xv - Hv' * Gv, 'fro')^2;

        % 项3：gamma * || Gv ||_F^2
        term3 = gamma * norm(Gv, 'fro')^2;

        obj = obj + term1 + term2 + term3;
    end
end

