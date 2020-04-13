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

% �o�䵹��ltheta(�v��)�H���ƪ���k
% �O���v�����d�򭭨�b(-�` ~ �`)����
% �`�b�o��ܪ��O�@�ӷ��p��,�ھڱоǳo��]��0.12
epsilon_init = 0.12;

% L_in�N��F��J�����g���Ӽ�,+1�O���Fbias
% L_out�h��ܿ�X�����g���Ӽ�
% rand(L_out, 1 + L_in),�o�ӫ��O�|�ͦ��@��L_out x (1 + L_in)���x�}
% �ӯx�}���U�����d��|�b0~1����
% �A��2�`���`,�N��N�x�}���Ӥ����d���ܬ��һݭn��-�` ~ �`
% �N�����F��ltheta���H����
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;






% =========================================================================

end
