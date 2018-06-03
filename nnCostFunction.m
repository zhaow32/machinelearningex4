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
%% feedforward propagation
a1=[ones(m,1) X];
z2=a1*Theta1';
a2=[ones(m,1) sigmoid(z2)];
z3=a2*Theta2';
a3=sigmoid(z3);
h=a3;

%% record labels
Y=zeros(m,num_labels);
I=eye(num_labels);
for i =1:m
    Y(i,:)=I(y(i),:);
end

%% calculate the penalty

pen=sum(sum(Theta1(:,2:end).^2,2))+sum(sum(Theta2(:,2:end).^2,2));

%% calculate loss
J=sum(sum(Y.*log(h)+(1-Y).*log(1-h)))/(-m) + lambda*pen/(2*m);

%% find sigma and delta
sigma3 = a3 - Y;
sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(m, 1) z2]);
sigma2 = sigma2(:, 2:end);

delta1=sigma2'*a1;
delta2=sigma3'*a2;

%% gradient
pen1=(lambda/m)* [zeros(size(Theta1, 1),1) Theta1(:,2:end)];
pen2=(lambda/m) * [zeros(size(Theta2, 1),1) Theta2(:,2:end)];
Theta1_grad=delta1./m + pen1;
Theta2_grad=delta2./m + pen2;
% ========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
