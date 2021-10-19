#This function implements Gradient Descend to fit the best parameters
function thethas=Gradient_Descend(X,y,alpha,thethas)
  m=length(y);
  thethas=thethas-(alpha/m)*(X(:,1)'*(sigmoid(X*thethas)-y));
  end