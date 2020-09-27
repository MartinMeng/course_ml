function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

delta_ys = X * theta - y;
J = 1/(2*m) * (transpose(delta_ys) * delta_ys) + lambda / (2*m) * (transpose(theta(2:end, :)) * theta(2:end, :));
% disp(size(J));

grad_0 = 1/m * (transpose(X(:, 1)) * delta_ys);
grad_rest = 1 / m * (transpose(X(:, 2:end)) * delta_ys) + lambda / m * theta(2:end, :);
grad = [grad_0; grad_rest];
% grad = [grad_0, grad_rest];
% disp(delta_ys);
% disp(theta);
% disp(X);
% temp = 1/m * (transpose(X(:, 1)) * delta_ys);
% disp(temp);
% theta_zero = theta;
% theta_zero(1) = 0;
% grad = 1/m * (transpose(X) * delta_ys) + lambda / m * theta_zero;
% =========================================================================

grad = grad(:);

end
