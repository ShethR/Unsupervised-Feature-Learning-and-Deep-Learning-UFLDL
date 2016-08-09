function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
Nv=size(data,2); % # of samples
% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

a1=data;

Z=W1*a1;
Z2=bsxfun(@plus,Z,b1);
a2=sigmoid(Z2);
clear Z;

Z=W2*a2;
Z3=bsxfun(@plus,Z,b2);
a3=sigmoid(Z3);

rowHat=sum(a2,2)/Nv;
row=repmat(sparsityParam,size(rowHat));

% Cost Function
% For more description of formulae please see 
  % http://nlp.stanford.edu/~socherr/sparseAutoencoder_2011new.pdf
  
sparsityCost=beta.*(sum(sparsityParam.*(log(row./rowHat)) + ((1-row).*(log((1-row)./(1-rowHat))))));
regCost=(lambda/2).*(sum(sum(W1.^2))+sum(sum(W2.^2)));
cost=sum(sum(((a3-data).^2)./(2*Nv))) + sparsityCost + regCost;

% Backpropogation

da3=a3.*(1-a3);
delta3 = -(data-a3).*da3;

da2=a2.*(1-a2);
sparsityDelta=repmat(beta*((-row./rowHat)+((1-row)./(1-rowHat))),[1,Nv]);
delta2=(W2'*delta3 + sparsityDelta).* (da2);


W1grad=(delta2*a1')/Nv + lambda.*W1;
b1grad=sum(delta2,2)/Nv;

W2grad=(delta3*a2')/Nv' + lambda.*W2;
b2grad=sum(delta3,2)/Nv;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

