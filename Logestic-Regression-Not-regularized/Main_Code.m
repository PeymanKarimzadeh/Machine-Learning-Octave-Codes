#This program is writen in order to implement logestic regression for a classification problem

#The first part is receiving data and implementing X and y
input_data=input("Please type the file name of your data:\n");
Data=load(input_data);
X=Data(:,[1:2]);
y=Data(:,3);

#The second part is plotting the data
x_label=input("Please type your favorable name of your X axis\n");
y_label=input("Please type your favorable name of your Y axis\n");
#Note that we call y=1 to posetive and y=0 to negetive
figure; hold on;
pos=find(y==1); neg=find(y==0);
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);
hold off;
xlabel(x_label)
ylabel(y_label)
legend('Admitted','Not Admitted')
fprintf("The program is paused, Press Enter to continue\n");
pause;
#The third part is testing sigmoid function
#Note that sigmoid function in z=0 is equal to 0.5
test_sigmoid=sigmoid(0);
fprintf("This is a test for sigmoid function, so the right answer should be 0.5, The test is equal to:\n")
disp(test_sigmoid);
fprintf("The program is paused, Press Enter to continue\n");
pause;

#The next part is implementing Gradient Descend in order to fit the best parameters
#At first, the cost function will be calculated
Design_matrix=[ones(size(y)) X];
[p,q]=size(Design_matrix);
initial_theta=zeros(q,1);
cost_function=costFunction(Design_matrix,y,initial_theta);
fprintf("\n The cost calculated by initial guess of theta is equal to:\n");
disp(cost_function);
#The next step in this part is implementing Gradiend Descend
alpha=0.01;
the_best_theta=Gradient_Descend(Design_matrix,y,alpha,initial_theta);
fprintf("The best parameters that are found by Gradient_Descend is equal to:\n");
disp(the_best_theta);

#The next part is implementing fminunc to fit the best parameters
options=optimset('GradObj','on','MaxIter',400);
[theta, cost] =fminunc(@(t)(cost_function_fminunc(t, Design_matrix, y)), initial_theta, options);
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('Expected cost (approx): 0.203\n');
fprintf('theta that found by fminunc is equal to: \n');
fprintf(' %f \n', theta);
fprintf('Expected theta (approx):\n');
fprintf(' -25.161\n 0.206\n 0.201\n');
fprintf("The program is paused, Press Enter to continue\n");
pause;

#This part of code is writen in order to predict wheter the student is Admitted or not
grade_one=(input("Insert the grade one:\n"));
grade_two=(input("Insert the grade two:\n"));
predicred=predict(grade_one,grade_two,theta);
fprintf("This student:\n")
disp(predicred);

#This part of code plots the Decistion Boundary line (Tha line with the best theta)
hold on;
plotDecisionBoundary(theta, Design_matrix, y);
hold off;
