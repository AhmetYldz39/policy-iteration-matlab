function r = reward(rewardstate_idx, rewardaction_idx, cost, s_idx, a)

x = find(rewardstate_idx == s_idx, 1);
if ~isempty(x)
    % to find the index
    i1 = mod(x, numel(rewardstate_idx(:, 1)));  % row index
    if i1 == 0
        i1 = numel(rewardstate_idx(:, 1));
    end
    i2 = ceil(x / numel(rewardstate_idx(:, 1)));  % column index
    if (a == rewardaction_idx(i1, i2))
        if cost(rewardstate_idx(i1, i2)) == 0
            r = 1e8;
        else
            r = 1/cost(rewardstate_idx(i1,i2));
        end
    else
        r = 0;
    end
else
    r = 0;
end

end