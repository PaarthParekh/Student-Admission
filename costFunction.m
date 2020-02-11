function [J, grad] = costFunction(theta, X, y)
m = length(y);
J = 0;
grad = zeros(size(theta));
pred=sigmoid(X*theta);
sub=1-pred;
error1=(-y.*log(pred))-((1-y).*log(sub));
J=1/m*(sum(error1));
grad=1/m*(sum((pred-y).*X));







% =============================================================

end
