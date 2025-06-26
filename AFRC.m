function [obj_value,G] = AFRC(X, m_v, maxIter, tol,lambda,gamma)
%
% 输入:
%   X      - 包含 V 个视图的 cell 数组，每个 X{v} 为 n×d_v 的数据矩阵
%   m_v    - 长度为 V 的向量，第 v 个元素为视图 v 的锚点数 m_v(v),对于特征数量超过10000的数据集，建议锚点数量小于等于6%。
%   maxIter- 最大迭代次数
%   tol    - 收敛阈值
%
% 输出:
%   H      - 包含 V 个视图锚表示的 cell 数组，每个 H{v} 为 m_v(v)×n
%   G      - 包含 V 个视图锚图的 cell 数组，每个 G{v} 为 m_v(v)×d_v
%   W      - 包含 V 个视图投影矩阵的 cell 数组，每个 W{v} 为 m×m_v(v)
%   H_star - m×n 的一致性锚表示矩阵
%   alpha  - V×1 视图权重向量，这里都设置为每个视图权重都设置为1
    if nargin < 3
        maxIter = 100;
    end
    if nargin < 4
        tol = 1e-3;
    end

    V = numel(X);                   % 视图数
    n = size(X{1}, 1);              % 样本数
    d_v = zeros(V,1);               % 各视图特征维度
    for v = 1:V
        [nx, dv] = size(X{v});
        assert(nx == n, '所有视图的样本数 n 必须相同');
        d_v(v) = dv;
    end

    m = sum(m_v);                   % 总锚点数（所有视图之和）
    alpha = ones(V,1);          % 初始化视图权重为均匀分布

    % 初始化变量 H_v, G_v, W_v, H_star
    H = cell(V,1);
    G = cell(V,1);
    W = cell(V,1);
    % 随机初始化 H_v（非负随机值）
    for v = 1:V
        H{v} = rand(m_v(v), n);
    end
    % 随机初始化 G_v，每列非负且和为1
    for v = 1:V
        Gv = rand(m_v(v), d_v(v));
        Gv = bsxfun(@rdivide, Gv, sum(Gv,1));  % 列归一化
        G{v} = Gv;
    end
    % 随机初始化 W_v 为行正交矩阵（m×m_v）
    for v = 1:V
        % 生成一个 m×m_v 的随机矩阵，然后 QR 分解取 Q
        temp = rand(m, m_v(v));
        [Q, ~] = qr(temp, 0);     % Q 为 m×m_v，列正交
        W{v} = Q;
    end
    % 随机初始化 H_star
    H_star = rand(m, n);
    iter=1;
    obj_value(iter) = compute_objective(X, H, W, G, H_star, alpha, lambda, gamma);
    % 迭代优化
 for iter = 2:maxIter
    H_prev = H_star;

    % 2. 更新 W_v（采用 SVD，确保正交）
    for v = 1:V
        Av = H_star * H{v}';   % m×m_v
        [U, ~, Vt] = svd(Av, 'econ');
        W{v} = U * Vt';
        
        % 清除不再使用的临时变量
        clear Av U Vt
    end

    % 3. 更新 H_v 的闭式解
    for v = 1:V
        Wv = W{v}; Gv = G{v};
        A = alpha(v)^2 * (Wv' * Wv) + lambda * (Gv * Gv');
        B = alpha(v)^2 * (Wv' * H_star) + lambda * (Gv * X{v}');

        H{v} = pinv(A) * B;

        % 清除临时变量
        clear A B Wv Gv
    end

    % 4. 更新 G_v：对每列用 quadprog 求解 QP
    options = optimoptions('quadprog','Display','off');
    for v = 1:V
        Hv = H{v}; 
        Q = lambda * (Hv * Hv') + gamma * eye(m_v(v)); 
        Hquad = 2 * Q; 
        Gv = zeros(m_v(v), d_v(v));
        Aeq = ones(1, m_v(v));
        beq = 1;
        lb = zeros(m_v(v), 1);
        
        for j = 1:d_v(v)
            xj = X{v}(:, j);  
            f = -2 * lambda * (Hv * xj);
            Gv(:, j) = quadprog(Hquad, f, [], [], Aeq, beq, lb, [], [], options); 
        end

        G{v} = Gv;

        % 清除临时变量
        clear Q Hquad Gv Hv Aeq beq lb
    end

    % 5. 更新 H_star（加权平均）
    a_sum = alpha' * alpha;
    H_star = zeros(m, n);
    parfor v = 1:V
        H_star = H_star + alpha(v)^2 * W{v} * H{v};
    end 
    H_star = (1 / a_sum) * H_star;

    obj_value(iter) = compute_objective(X, H, W, G, H_star, alpha, lambda, gamma);
    tmp = abs((obj_value(iter) - obj_value(iter - 1)) / obj_value(iter - 1));
    
    if tmp < tol 
        fprintf('迭代于第 %d 次收敛，停止优化。\n', iter);
        break;
    end

    % 清除每轮结束后不再使用的变量
    clear tmp a_sum
end
