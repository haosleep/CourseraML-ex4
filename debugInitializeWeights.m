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

% �qex4.m��part6�i�����g�����Ϊ���l�v�����ӭn�H���B�z
% ���p�G�����H������,�C���ͦ��ɨ��o���ȳ��|���P
% �o��Q��sin���,�]�����O����0,�i�H�Ψӷ��l�v��
% �ӥB�C�������|�O�P�˪��x�},�ܫK��debug
W = reshape(sin(1:numel(W)), size(W)) / 10;

% =========================================================================

end
