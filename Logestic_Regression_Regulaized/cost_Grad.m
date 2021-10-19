#This code will return cost function and Gradiants
function [Jval,Grad]=cost_Grad(theta,X,y,lambda)
  m=length(y);
  Grad=zeros(size(theta));
  Jval=sum((-y.*log(sigmoid(X*theta)))-(1-y).*log(1-sigmoid(X*theta)))/(m) +(0.5*lambda/m)*sum(theta(2:end).^2);
  Grad(1)=(X(:,1)'*(sigmoid(X*theta)-y))/m;
  [p,q]=size(theta);
  for i=2:p
    Grad(i)=(X(:,i)'*(sigmoid(X*theta)-y))/m + (lambda/m)*theta(i);
  endfor
end