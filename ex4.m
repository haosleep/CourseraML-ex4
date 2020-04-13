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

% 本次作業主要是練習神經網路的損失函數和梯度的求法(反向傳播算法)和計算
% 使用的是和前次作業一樣的5000x400手寫辨識資料來進行測試
load('ex4data1.mat');
m = size(X, 1);

% Randomly select 100 data points to display
% 這邊也跟前次作業相同
% 利用displayData.m將隨機100項資料的圖片print出
% 用以確認ex4data1.mat是否有正確讀取
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

% 此次也一樣已經先準備好了各層的theta(權重)
% Theta1是25x401的矩陣
% Theta2是10x26的矩陣
load('ex4weights.mat');

% Unroll parameters 

% 跟上次不同的是,這次要練習將所有theta的資料結合成一個變數
% 用以方便傳值
% 需要用時再利用reshape拆解回原本的矩陣
% 結合後,nn_params格式會是10285x1
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

% 開始處理本次作業重點nnCostFunction.m
% 因為這部分略微複雜,作業設計成可以分成三個階段來完成

% 這邊先將lambda設為0,讓結果可以先無視正規化
% 且只先回傳J(損失函數),可以先將梯度的部分省略
% 主要先完成損失函數的計算(part3作業)
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

% 這邊驗證損失函數的結果
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

% 接著lambda設為1,不過同樣只回傳J
% 要在nnCostFunction.m的損失函數加上正規化的部分(part4作業)
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

% 驗證完整的損失函數結果
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

% 接下來要開始完成nnCostFunction.m利用反向傳播算法求梯度的部分
% 而在這之前,先在sigmoidGradient.m中完成對g(z)的微分 (part5作業)
% 這在反向傳播算法時會用到,所以先將這函數完成會比較方便
% 以下是利用[-1 -0.5 0 0.5 1]來驗證sigmoidGradient結果是否正確
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

% 先前線性回歸,邏輯回歸等作業中
% 會將一開始的theta全設為0,再通過梯度下降等方式迭代後求得適當的theta
% 但在神經網路中若把一開始的theta都設為相同值
% 會導致隱藏層的每個神經元的結果一致,不論經過幾次迭代訓練都是在計算相同的函式
% 這稱之為對稱性破缺(symmetry breaking)的問題,會讓梯度下降失去作用
% 因此,這邊要利用randInitializeWeights.m
% 對初始的theta值不再是全設為0,而是用隨機化給值的方式處理(part6作業)
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

% 到這邊,必須要利用反向傳播算法將nnCostFunction.m完成求梯度的部分(part7作業)

fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients

% 在開始進行機器學習訓練之前,先進行梯度檢查,用以確保反向傳播算法求得的梯度下降算法是準確的
% 梯度檢查是在每次迭代時的θ,再取兩個相鄰的θ - ε,θ + ε,ε是大於0的極小值
% (f(θ + ε) - f(θ - ε))/2ε 的值就會極接近θ的梯度
% 但也因為一個θ要附帶兩個θ - ε,θ + ε進行檢查
% 如果邊訓練一邊檢查的話,訓練過程會變得極度緩慢
% 因此梯度檢查只用來確認有沒有BUG,在訓練前先檢查較為妥當

% 這邊先不給checkNNGradients.m參數
% 在checkNNGradients.m裡面的寫法中,沒給參數的呼叫會將lambda設為0,也就是忽視掉正規化
% 如果這邊梯度檢查發現有問題的話,錯誤的部分就跟正規化無關
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients

% 這次加上了正規化的參數再次進行梯度檢查
% 如果part7時沒問題,這邊檢查卻發現有問題的話
% BUG就極可能是出現在正規化相關的部分
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
% 因為lambda值改動(和part4不同)
% 所以再次執行nnCostFunction.m,再一次的驗證結果是否正確
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

% 完成了算法nnCostFunction.m也經過梯度檢查後
% 接下來就可以進行機器學習來進行訓練
% 用的是前次作業(ex3)時也用過的fmincg

fprintf('\nTraining Neural Network... \n')

% 這邊可以嘗試改動迭代次數和拉格朗日參數
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
% 注意這邊用的不再是從ex4weights.mat得到的nn_params(part2的部分)
% 而是要用part6時取得的初始隨機權重initial_nn_params來正式進行機器學習,得出最後的theta值(nn_params)
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

% 這邊將學習後的Theta1用圖片展示出來
% 來觀察經過訓練後隱藏層對於給予資料的權重分配
displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

% 最後驗證機器學習的準確度
pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


