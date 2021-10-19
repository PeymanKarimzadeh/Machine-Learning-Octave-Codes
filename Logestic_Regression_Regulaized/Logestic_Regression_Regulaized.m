#In this program, we will implement regularized logestic regression to predict
#Note that your inseted data should just have 2 feature
#You can also use the other format but you need to change the code

# At first the data will be loaded
input_data=input("Please type the name of your file data:\n");
data=load(input_data);
X=data(:,[1:2]);
y=data(:,3);

# The next part is to plot the data
x_label=input(" Please type the label of x-axis:\n");
y_label=input(" Please type the label of y-axis:\n");
pos=find(y==1);
neg=find(y==0);
figure;
hold on;
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'r','MarkerSize', 7);
xlabel(x_label);
ylabel(y_label);
legend('Pass','Not Pass');
hold off;

# The next Part is to define initial guess for theta 
X1=X(:,1);
X2=X(:,2);
degree=input("Please Type the degree of your determined polynominal:\n");
X=mapFeature(X1,X2,degree);
[m,n]=size(X);
initial_theta=zeros(size(X,2),1);

# The next step is to train cost Function 
lambda=input("Insert an approperiat lambda:\n");
MaxIteration=input("Insert the Maximum number of Iterations:\n");
[cost,grad]=cost_Grad(initial_theta,X,y,lambda);

% Compute and display cost and gradient
% with all-ones theta and lambda = 10
test_theta = ones(size(X,2),1);
[cost, grad] = cost_Grad(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

option=optimset('GradObj','on','MaxIter',MaxIteration);
[theta, J, exit_flag] = ...
	fminunc(@(t)(cost_Grad(t, X, y, lambda)), initial_theta, option);
% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');
