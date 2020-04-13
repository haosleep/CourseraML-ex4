%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data

fprintf('Loading and Visualizing Data ...\n')

% �����@�~�D�n�O�m�߯��g�������l����ƩM��ת��D�k(�ϦV�Ǽ���k)�M�p��
% �ϥΪ��O�M�e���@�~�@�˪�5000x400��g���Ѹ�ƨӶi�����
load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
% �o��]��e���@�~�ۦP
% �Q��displayData.m�N�H��100����ƪ��Ϥ�print�X
% �ΥH�T�{ex4data1.mat�O�_�����TŪ��
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 2: Loading Parameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2

% �����]�@�ˤw�g���ǳƦn�F�U�h��theta(�v��)
% Theta1�O25x401���x�}
% Theta2�O10x26���x�}
load('ex4weights.mat');

% Unroll parameters 

% ��W�����P���O,�o���n�m�߱N�Ҧ�theta����Ƶ��X���@���ܼ�
% �ΥH��K�ǭ�
% �ݭn�ήɦA�Q��reshape��Ѧ^�쥻���x�}
% ���X��,nn_params�榡�|�O10285x1
% (25x401) + (10x26) = 10285
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Compute Cost (Feedforward) ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network that returns the cost only. You
%  should complete the code in nnCostFunction.m to return cost. After
%  implementing the feedforward to compute the cost, you can verify that
%  your implementation is correct by verifying that you get the same cost
%  as us for the fixed debugging parameters.
%
%  We suggest implementing the feedforward cost *without* regularization
%  first so that it will be easier for you to debug. Later, in part 4, you
%  will get to implement the regularized cost.
%
fprintf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

% �}�l�B�z�����@�~���InnCostFunction.m
% �]���o�������L����,�@�~�]�p���i�H�����T�Ӷ��q�ӧ���

% �o����Nlambda�]��0,�����G�i�H���L�����W��
% �B�u���^��J(�l�����),�i�H���N��ת������ٲ�
% �D�n�������l����ƪ��p��(part3�@�~)
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

% �o�����ҷl����ƪ����G
fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.287629)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =============== Part 4: Implement Regularization ===============
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.
%

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

% ����lambda�]��1,���L�P�˥u�^��J
% �n�bnnCostFunction.m���l����ƥ[�W���W�ƪ�����(part4�@�~)
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

% ���ҧ��㪺�l����Ƶ��G
fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n(this value should be about 0.383770)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 5: Sigmoid Gradient  ================
%  Before you start implementing the neural network, you will first
%  implement the gradient for the sigmoid function. You should complete the
%  code in the sigmoidGradient.m file.
%

fprintf('\nEvaluating sigmoid gradient...\n')

% ���U�ӭn�}�l����nnCostFunction.m�Q�ΤϦV�Ǽ���k�D��ת�����
% �Ӧb�o���e,���bsigmoidGradient.m��������g(z)���L�� (part5�@�~)
% �o�b�ϦV�Ǽ���k�ɷ|�Ψ�,�ҥH���N�o��Ƨ����|�����K
% �H�U�O�Q��[-1 -0.5 0 0.5 1]������sigmoidGradient���G�O�_���T
g = sigmoidGradient([-1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

% ���e�u�ʦ^�k,�޿�^�k���@�~��
% �|�N�@�}�l��theta���]��0,�A�q�L��פU�����覡���N��D�o�A��theta
% ���b���g�������Y��@�}�l��theta���]���ۦP��
% �|�ɭP���üh���C�ӯ��g�������G�@�P,���׸g�L�X�����N�V�m���O�b�p��ۦP���禡
% �o�٤�����٩ʯ}��(symmetry breaking)�����D,�|����פU�����h�@��
% �]��,�o��n�Q��randInitializeWeights.m
% ���l��theta�Ȥ��A�O���]��0,�ӬO���H���Ƶ��Ȫ��覡�B�z(part6�@�~)
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%

% ��o��,�����n�Q�ΤϦV�Ǽ���k�NnnCostFunction.m�����D��ת�����(part7�@�~)

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients

% �b�}�l�i������ǲ߰V�m���e,���i�����ˬd,�ΥH�T�O�ϦV�Ǽ���k�D�o����פU����k�O�ǽT��
% ����ˬd�O�b�C�����N�ɪ��c,�A����Ӭ۾F���c - �`,�c + �`,�`�O�j��0�����p��
% (f(�c + �`) - f(�c - �`))/2�` ���ȴN�|������c�����
% ���]�]���@�ӣc�n���a��ӣc - �`,�c + �`�i���ˬd
% �p�G��V�m�@���ˬd����,�V�m�L�{�|�ܱo���׽w�C
% �]������ˬd�u�ΨӽT�{���S��BUG,�b�V�m�e���ˬd��������

% �o�������checkNNGradients.m�Ѽ�
% �bcheckNNGradients.m�̭����g�k��,�S���Ѽƪ��I�s�|�Nlambda�]��0,�]�N�O���������W��
% �p�G�o�����ˬd�o�{�����D����,���~�������N�򥿳W�ƵL��
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients

% �o���[�W�F���W�ƪ��ѼƦA���i�����ˬd
% �p�Gpart7�ɨS���D,�o���ˬd�o�o�{�����D����
% BUG�N���i��O�X�{�b���W�Ƭ���������
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
% �]��lambda�ȧ��(�Mpart4���P)
% �ҥH�A������nnCostFunction.m,�A�@�������ҵ��G�O�_���T
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%

% �����F��knnCostFunction.m�]�g�L����ˬd��
% ���U�ӴN�i�H�i������ǲߨӶi��V�m
% �Ϊ��O�e���@�~(ex3)�ɤ]�ιL��fmincg

fprintf('\nTraining Neural Network... \n')

% �o��i�H���է�ʭ��N���ƩM�Ԯ�Ԥ�Ѽ�
%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
% �`�N�o��Ϊ����A�O�qex4weights.mat�o�쪺nn_params(part2������)
% �ӬO�n��part6�ɨ��o����l�H���v��initial_nn_params�ӥ����i������ǲ�,�o�X�̫᪺theta��(nn_params)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

% �o��N�ǲ᪺߫Theta1�ιϤ��i�ܥX��
% ���[��g�L�V�m�����üh��󵹤���ƪ��v�����t
displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

% �̫����Ҿ����ǲߪ��ǽT��
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


