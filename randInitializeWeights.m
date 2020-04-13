function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%


% Randomly initialize the weights to small values

% 這邊給初始theta(權重)隨機化的方法
% 是讓權重的範圍限制在(-ε ~ ε)之間
% ε在這表示的是一個極小值,根據教學這邊設為0.12
epsilon_init = 0.12;

% L_in代表了輸入的神經元個數,+1是為了bias
% L_out則表示輸出的神經元個數
% rand(L_out, 1 + L_in),這個指令會生成一個L_out x (1 + L_in)的矩陣
% 而矩陣的各元素範圍會在0~1之間
% 再乘2ε後減ε,就能將矩陣的個元素範圍變為所需要的-ε ~ ε
% 就完成了初始theta的隨機化
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;






% =========================================================================

end
