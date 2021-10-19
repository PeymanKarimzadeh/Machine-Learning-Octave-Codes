#This is costFunction code
function J=costFunction(X,y,theta)
  h_theta=X*theta;
  [m,n]=size(y);
  J=sum(y.*log(sigmoid(h_theta))+(1-y).*log(1-sigmoid(h_theta)))/(-m);
end