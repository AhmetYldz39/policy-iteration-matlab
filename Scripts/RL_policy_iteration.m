%% Dynamic Programming - Policy Iteration
% ------------------------------------------------
% Dynamic programming is used.
% Policy iteration method is used.
% ------------------------------------------------
%% User Inputs
eps = 1e-3;  % tolerance value
gamma = 0.9;  % discount rate
a_step = 0.1;  % change in value for each action
bd = [-5, 5]';  % boundaries for the states
max_iter = 1000;  % maximum iteration number for policy iteration
fun = 'terrain_shape';  % function used in the algorithm
nvar = nargin(fun);  % variable number of the function

%% All Possible States
X = (bd(1):a_step:bd(2))';  % variables can take value in this interval
C1 = numel(X)*numel(X);  % combination to get the size of state space
S = zeros(C1,1);        % state space
for i = 1:numel(X)
    s1 = (i-1)*numel(X)+1;
    s2 = i*numel(X);
    S(s1:s2,1) = X(i);
    for j = 1:numel(X)
        S(s1+j-1,2) = X(j);
    end
end
clear s1 s2

%% All Possible Actions
U = [-a_step;0;a_step];  % possible actions
C2 = numel(U)*numel(U);  % number of all possible action combinations
A = zeros(C2, nvar);  % action space
for i = 1:numel(U)
    a1 = (i - 1) * numel(U) + 1;
    a2 = i * numel(U);
    A(a1:a2, 1) = U(i);
    for j = 1:numel(U)
        A(a1+j-1, 2) = U(j);
    end
end
clear a1 a2

%% Obtaining Rewarded State-Action Pairs
[r,~] = size(S); 
cost = zeros(r,1);
for i = 1:r
    cost(i) = terrain_shape(S(i, 1), S(i, 2));
end

R = zeros(r,1); % rewards for each state will be written in this array
mins = zeros(r,1); % local/global minimum points will be taking the value of 1
for i = 1:r
    if ~(ismember(S(i,1), bd) || ismember(S(i,2), bd))
        check = 0;  % to check the state if it is a local/global minimum
        for j = 1:size(A,1)
            temp1 = S(i,1) + A(j,1);
            temp2 = S(i,2) + A(j,2);
            if cost(i) > RB_func(temp1,temp2) % if it is not a minimum
                check = check + 1;
            end
        end

        if abs(check) < 1e-8 % this means it is a minimum point - terminal state
            mins(i) = 1;
        end
    end
end
mins_idx = find(abs(mins - 1) < 1e-8);

% for each minimum point, there are nine states that can reach the minimum
% with just one action (including the minimum state itself)
rewardstate_idx  = zeros(numel(mins_idx), size(A,1));
rewardaction_idx = zeros(numel(mins_idx), size(A,1));
for i = 1:numel(mins_idx)
    for j = 1:size(A, 1)
        x1 = S(mins_idx(i), 1) + A(j, 1);
        x2 = S(mins_idx(i), 2) + A(j, 2);
        y1 = find(abs(S(:, 1) - x1) < 1e-8);
        y2 = find(abs(S(:, 2) - x2) < 1e-8);
        z  = intersect(y1, y2);
        rewardstate_idx(i, size(A,1)+1-j)  = z;
        rewardaction_idx(i, size(A,1)+1-j) = size(A, 1) + 1 - j;
    end
end
clear temp1 temp2 check x1 x2 y1 y2 z

%% Policy Iteration - Bellman Equation
Q = zeros(numel(S(:,1)),numel(A(:,1)));  % initialization of Q-value of each state-action pairs
Q_old = Q;  % to execute policy improvement
h = a_step*ones(numel(S(:,1)),2);  % initialization of policy
h_old = h;  % to check if policy is improved
PI_count = 0;  % to count policy improvements
for i = 1:max_iter
    fprintf("Iteration: %d\n", i);
    h(1,:) = [ a_step,  a_step];
    h(numel(S(:, 1)), :) = [-a_step, -a_step];
    
    % Policy Evaluation
    for j = 1:numel(S(:,1))
        x = S(j,:);  % current state
        for k = 1:numel(A(:,1))
            u = A(k,:);  % current action
            if isempty(find(mins_idx == x, 1)) % if the state is not a terminal state
                r      = reward(rewardstate_idx, rewardaction_idx, cost, j, k);

                xn = x + u;  % new state after the action
                if xn(1) < bd(1)
                    xn(1) = bd(1);
                elseif xn(1) > bd(2)
                    xn(1) = bd(2);
                end
                if xn(2) < bd(1)
                    xn(2) = bd(1);
                elseif xn(2) > bd(2)
                    xn(2) = bd(2);
                end

                y3 = find(abs(S(:, 1) - xn(1)) < 1e-8);
                y4 = find(abs(S(:, 2) - xn(2)) < 1e-8);
                xn_idx = intersect(y3, y4);  % new state index

                un = NextAction(h, j);  % next action according to policy
                y5 = find(abs(A(:, 1) - un(1)) < 1e-8);
                y6 = find(abs(A(:, 2) - un(2)) < 1e-8);
                un_idx = intersect(y5, y6);  % new action index

                qn = r + gamma*Q_Value(Q, xn_idx, un_idx);
                Q = Q_Updated(Q, qn, j, k);
            end
        end
    end

    % Policy Improvement
    if abs(sum(sum(Q - Q_old))) < eps
        h = NewPolicy(Q, A);  % updating the policy
        PI_count = PI_count + 1;  % counting policy improvements
        if abs(sum(h - h_old)) < eps
            h_optimal = h;
            IsOptimal = true;

            % save the optimal result
            PI = struct();
            PI.Q = Q;
            PI.h = h;
            PI.PI_count = PI_count;
            filename = ['PI_', num2str(i)]; 
            save(filename, 'PI', '-v7.3') 
            fprintf("Process has been finished at iteration %d\n", i);

            break;
        else
            % setting policy and Q-values to continue the loop
            h_old = h;
            Q = zeros(numel(S(:, 1)), numel(A(:, 1)));
        end
    else
        Q_old = Q;
    end

    if abs(max_iter - i) < 1e-8
        % save the latest result
        PI = struct();
        PI.Q = Q;
        PI.h = h;
        PI.PI_count = PI_count;
        filename = ['PI_', num2str(i)]; 
        save(filename, 'PI', '-v7.3')     
    end
end


%% Functions

% obtaining next action according to policy
function un = NextAction(h,s_idx)
% h = policy
% s_idx = state number
un = h(s_idx,:);
end

% obtaining Q-value of the state-action pairs
function q = Q_Value(Q,xn_idx,un_idx)
% Q = current Q-values
% xn_idx = state index
% un_idx = action index
q = Q(xn_idx,un_idx);
end

% obtaining updated Q-values
function Q = Q_Updated(Q,qn,xn_idx,un_idx)
% Q = current Q-values
% qn = new Q-value
% xn_idx = state index
% un_idx = action index
Q(xn_idx, un_idx) = qn;
end

% updating the policy
function H = NewPolicy(Q,A)
[~,idx] = max(Q,[],2);  % arg max values for each state respect to action
H = zeros(numel(idx),2);
for i = 1:numel(idx)
    H(i, :) = A(idx(i), :); % by using action indices, policy is updated
end
end
