function prob = predict()
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)




% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%
[file,folder]=uigetfile;
filename=fullfile(folder,file);
data=load(filename);
X=data(:,1:2);
y=data(:,3);
[m, n] = size(X);
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
fprintf('Cost at theta found by fminunc: %f\n', cost);
plotBoundary(theta, X, y);
str = inputdlg('Enter the scores of exam 1 and 2 separated by spaces or commas');
numbers = str2num(str{1});
x=reshape(numbers,[1,2]);
x=[1,x];
m = size(X, 1); % Number of training examples
% You need to return the following variables correctly
p = zeros(m, 1);

pred1=sigmoid(X*theta);
p=pred1>=0.5;
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
prob = sigmoid(x * theta);
fprintf('For this student, we predict an admission probability of %f\n\n', prob);
pred=sigmoid(x*theta);
prob=(pred>=0.5);





% =========================================================================


end
