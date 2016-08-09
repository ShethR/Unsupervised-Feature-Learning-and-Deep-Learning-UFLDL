function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);
lamb=repmat(lambda,(size(theta)));
numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


z=theta*data;
m=max(z);
z=bsxfun(@minus, z,m);

num=exp(z);
den=sum(num);
a1=bsxfun(@rdivide,num,den);
a=log(a1);
cost=(sum(sum(groundTruth.*a))/-size(data,2)) + lambda/2 * (sum(sum(theta.^2)));
M=groundTruth-a1;
thetagrad=-(M*data')/size(data,2);
thetagrad = thetagrad + (lamb.*theta);
        
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

