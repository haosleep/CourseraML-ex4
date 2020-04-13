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

% 利用reshape將傳進來的nn_params(先前在ex4部分結合起來的全部theta值)整理回原本的Theta1,Theta2矩陣
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

% 傳進來的X格式是5000x400
% 神經網路的架構為一個隱藏層
% 輸入層400個神經元(+1 bias)
% 隱藏層25個神經元(+1 bias)
% 輸出層10個神經元

% 這邊的計算跟前次作業(ex3)的predict.m相同
% 照著公式推得X(a1),z2,a2,z3,a3即可
% X格式 5000x401
X = [ones(m, 1) X];
% z2格式 5000x401 * 401x25 = 5000x25
z2 = X * Theta1';
a2 = sigmoid(z2);
% a2格式 5000x26
a2 = [ones(m, 1) a2];
% z3格式 5000x26 * 26x10 = 5000x10
z3 = a2 * Theta2';
% a3格式 5000x10
a3 = sigmoid(z3);

% 先列出1~num_labels(10)的向量(列),格式會是1 x num_labels(10)
compare_y = [1:num_labels];
% 因為傳進來的y的格式是5000x1,只標記了結果為1~10的哪一種
% 利用(y == compare_y)的方式,可將其攤開為5000x10 (跟a3格式相同)
% 各列會是非0即1的比對結果
yk = (y == compare_y);
% 計算損失函數(尚未加正規化)
% 基本公式也一樣是 (-y*log(hx) - (1-y)*log(1-hx)) ,全部總和後平均
% 這邊要注意的是矩陣乘法的處理,因為y已經不再是向量
% 在先前只有0和1的分類問題下的邏輯回歸損失函數,y和預測結果的hx只是mx1的向量
% 因此可以利用矩陣乘法1xm * mx1 = 1x1,計算完後剛好就是總和結果,之後再除以m (以教學上的公式來說,可視為K = 1)
% 但這次手寫辨識結果是1~10(輸出層的神經元個數是10),也就是10組的0和1分類 => K = 10
% 在這個情況下,y和預測結果不再是mx1的向量,而是mx10
% 直接用矩陣乘法處理的話,得到的10x10的結果裡會包含對應錯的部分
% (以教學上的公式說明, y對應的hx,k必須相同
%  矩陣乘法的結果下只有對角線的部分才是相同的k下的y和hx的計算總和)
% 因此,在求總和時,要用diag(取矩陣的對角線轉為向量)
% 再用sum將向量加總
J = sum(diag(-yk' * log(a3) - (1 - yk)' * log(1 - a3))) / m;
% 除了上述的方法外,也可以在yk和log(hk)計算時使用.* ,然後再用兩個sum求出矩陣每個元素的總和亦可

% 完成上面的部分後,已經可以在ex4.m的part3階段得到正確的答案

% 接著處理正規化的部分
% 基本公式也是一樣,同樣θ0的部分不進行處理
% 因此要將Theta1,Theta2另存一個變數將第一行(對應θ0的部分)改為0
tempT1 = Theta1;
tempT1(:,1) = 0;
tempT2 = Theta2;
tempT2(:,1) = 0;

% 接著套入公式
% 這邊theta一樣,因為也已經不是向量的關係,不方便再用矩陣乘法來達到平方後總和的結果
% 這裡使用.^2的方式把每個元素取平方後,再用兩個sum求每個元素總和
% 後面再補乘上lambda / (2 * m)
J = J + (sum(sum(tempT1.^2)) + sum(sum(tempT2.^2))) * lambda / (2 * m);

% 完成上面的部分後,已經可以在ex4.m的part4階段得到正確的答案


% 接著要利用反向傳播算法求得梯度
% 也是就照著公式走,只要隨時注意矩陣格式

% 將最後的輸出層的值(a3)和實際結果值(yk)相減,得出δ3
% δ3格式是5000x10
delta3 = a3 - yk;

% 再計算δ2
% 除了最後一層外
% 其下的δi 公式是Θi' * δ(i+1) .* g'(zi) (教學上i是用l,但英文l跟數字1太相像不適合寫在程式註解)
% 因此這邊i代入2
% 求得Θ2' * δ3 .* g'(z2)
% 這邊要注意的是δ要對應神經元個數(隱藏層25個神經元)
% 而傳進來的Theta2包含了bias的θ值(格式是10x26)
% 所以利用(:,2:end)把給bias的θ值去掉 (也就是教學上所提的,去掉δ0)

% 照著公式處理,不過為了讓矩陣格式對應需要適度的將矩陣進行轉置
% δ2格式是 ((10x25)' * (5000x10)')' .* 5000x25
% => 5000x25
delta2 = (Theta2(:,2:end)' * delta3')' .* sigmoidGradient(z2);

% 因為沒有δ1,從輸出層δm求到最後δ2即可
% δ2求出後,就可計算最後的梯度
% 同樣正規化的部分不包含θ0,所以使用前面處理好的tempT1,2取代Theta1,2

% 格式 25x401
Theta1_grad = delta2' * X / m + (lambda/m) * tempT1;
% 格式 10x26
Theta2_grad = delta3' * a2 / m + (lambda/m) * tempT2;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
% 將算出的梯度整理回傳進來的格式後回傳
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
