function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% �Q��reshape�N�Ƕi�Ӫ�nn_params(���e�bex4�������X�_�Ӫ�����theta��)��z�^�쥻��Theta1,Theta2�x�}
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% �Ƕi�Ӫ�X�榡�O5000x400
% ���g�������[�c���@�����üh
% ��J�h400�ӯ��g��(+1 bias)
% ���üh25�ӯ��g��(+1 bias)
% ��X�h10�ӯ��g��

% �o�䪺�p���e���@�~(ex3)��predict.m�ۦP
% �ӵۤ������oX(a1),z2,a2,z3,a3�Y�i
% X�榡 5000x401
X = [ones(m, 1) X];
% z2�榡 5000x401 * 401x25 = 5000x25
z2 = X * Theta1';
a2 = sigmoid(z2);
% a2�榡 5000x26
a2 = [ones(m, 1) a2];
% z3�榡 5000x26 * 26x10 = 5000x10
z3 = a2 * Theta2';
% a3�榡 5000x10
a3 = sigmoid(z3);

% ���C�X1~num_labels(10)���V�q(�C),�榡�|�O1 x num_labels(10)
compare_y = [1:num_labels];
% �]���Ƕi�Ӫ�y���榡�O5000x1,�u�аO�F���G��1~10�����@��
% �Q��(y == compare_y)���覡,�i�N���u�}��5000x10 (��a3�榡�ۦP)
% �U�C�|�O�D0�Y1����ﵲ�G
yk = (y == compare_y);
% �p��l�����(�|���[���W��)
% �򥻤����]�@�ˬO (-y*log(hx) - (1-y)*log(1-hx)) ,�����`�M�ᥭ��
% �o��n�`�N���O�x�}���k���B�z,�]��y�w�g���A�O�V�q
% �b���e�u��0�M1���������D�U���޿�^�k�l�����,y�M�w�����G��hx�u�Omx1���V�q
% �]���i�H�Q�ίx�}���k1xm * mx1 = 1x1,�p�⧹���n�N�O�`�M���G,����A���Hm (�H�оǤW�������ӻ�,�i����K = 1)
% ���o����g���ѵ��G�O1~10(��X�h�����g���ӼƬO10),�]�N�O10�ժ�0�M1���� => K = 10
% �b�o�ӱ��p�U,y�M�w�����G���A�Omx1���V�q,�ӬOmx10
% �����ίx�}���k�B�z����,�o�쪺10x10�����G�̷|�]�t������������
% (�H�оǤW����������, y������hx,k�����ۦP
%  �x�}���k�����G�U�u���﨤�u�������~�O�ۦP��k�U��y�Mhx���p���`�M)
% �]��,�b�D�`�M��,�n��diag(���x�}���﨤�u�ର�V�q)
% �A��sum�N�V�q�[�`
J = sum(diag(-yk' * log(a3) - (1 - yk)' * log(1 - a3))) / m;
% ���F�W�z����k�~,�]�i�H�byk�Mlog(hk)�p��ɨϥ�.* ,�M��A�Ψ��sum�D�X�x�}�C�Ӥ������`�M��i

% �����W����������,�w�g�i�H�bex4.m��part3���q�o�쥿�T������

% ���۳B�z���W�ƪ�����
% �򥻤����]�O�@��,�P�ˣc0���������i��B�z
% �]���n�NTheta1,Theta2�t�s�@���ܼƱN�Ĥ@��(�����c0������)�אּ0
tempT1 = Theta1;
tempT1(:,1) = 0;
tempT2 = Theta2;
tempT2(:,1) = 0;

% ���ۮM�J����
% �o��theta�@��,�]���]�w�g���O�V�q�����Y,����K�A�ίx�}���k�ӹF�쥭����`�M�����G
% �o�̨ϥ�.^2���覡��C�Ӥ����������,�A�Ψ��sum�D�C�Ӥ����`�M
% �᭱�A�ɭ��Wlambda / (2 * m)
J = J + (sum(sum(tempT1.^2)) + sum(sum(tempT2.^2))) * lambda / (2 * m);

% �����W����������,�w�g�i�H�bex4.m��part4���q�o�쥿�T������


% ���ۭn�Q�ΤϦV�Ǽ���k�D�o���
% �]�O�N�ӵۤ�����,�u�n�H�ɪ`�N�x�}�榡

% �N�̫᪺��X�h����(a3)�M��ڵ��G��(yk)�۴�,�o�X�_3
% �_3�榡�O5000x10
delta3 = a3 - yk;

% �A�p��_2
% ���F�̫�@�h�~
% ��U���_i �����O�Ki' * �_(i+1) .* g'(zi) (�оǤWi�O��l,���^��l��Ʀr1�Ӭ۹����A�X�g�b�{������)
% �]���o��i�N�J2
% �D�o�K2' * �_3 .* g'(z2)
% �o��n�`�N���O�_�n�������g���Ӽ�(���üh25�ӯ��g��)
% �ӶǶi�Ӫ�Theta2�]�t�Fbias���c��(�榡�O10x26)
% �ҥH�Q��(:,2:end)�⵹bias���c�ȥh�� (�]�N�O�оǤW�Ҵ���,�h���_0)

% �ӵۤ����B�z,���L���F���x�}�榡�����ݭn�A�ת��N�x�}�i����m
% �_2�榡�O ((10x25)' * (5000x10)')' .* 5000x25
% => 5000x25
delta2 = (Theta2(:,2:end)' * delta3')' .* sigmoidGradient(z2);

% �]���S���_1,�q��X�h�_m�D��̫�_2�Y�i
% �_2�D�X��,�N�i�p��̫᪺���
% �P�˥��W�ƪ��������]�t�c0,�ҥH�ϥΫe���B�z�n��tempT1,2���NTheta1,2

% �榡 25x401
Theta1_grad = delta2' * X / m + (lambda/m) * tempT1;
% �榡 10x26
Theta2_grad = delta3' * a2 / m + (lambda/m) * tempT2;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
% �N��X����׾�z�^�Ƕi�Ӫ��榡��^��
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
