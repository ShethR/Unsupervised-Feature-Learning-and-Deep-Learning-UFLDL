function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
depth=numel(stack)+1;



activation=cell(depth,1);
activation{1}=data;
for i=1:numel(stack)
    z = bsxfun(@plus, (stack{i}.w * activation{i}), stack{i}.b);
    activation{i+1} = sigmoid(z);
end

z=softmaxTheta*activation{depth};
m=max(z);
z=bsxfun(@minus, z,m);
num=exp(z);
den=sum(num);
a=bsxfun(@rdivide,num,den);
[~,pred]=max(a);



% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
