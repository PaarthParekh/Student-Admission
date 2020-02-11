function plotBoundary(theta, X, y)
plotData(X(:,2:3), y);
hold on
% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));
% Plot, and adjust axes for better viewing
plot(plot_x, plot_y)
% Legend, specific for the exercise
legend('Admitted', 'Not admitted', 'Decision Boundary')
axis([30, 100, 30, 100])