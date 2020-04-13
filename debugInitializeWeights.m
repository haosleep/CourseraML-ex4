function W = debugInitializeWeights(fan_out, fan_in)
%DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
%incoming connections and fan_out outgoing connections using a fixed
%strategy, this will help you later in debugging
%   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
%   of a layer with fan_in incoming connections and fan_out outgoing 
%   connections using a fix set of values
%
%   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
%   the first row of W handles the "bias" terms
%

% Set W to zeros
W = zeros(fan_out, 1 + fan_in);

% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging

% 從ex4.m的part6可知神經網路用的初始權重應該要隨機處理
% 但如果直接隨機給值,每次生成時取得的值都會不同
% 這邊利用sin函數,因為不是全為0,可以用來當初始權重
% 而且每次取都會是同樣的矩陣,很便於debug
W = reshape(sin(1:numel(W)), size(W)) / 10;

% =========================================================================

end
