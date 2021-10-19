function [Jval,Gradient]=cost_function_fminunc(theta,X,y)
  Gradient=zeros(size(theta));
  h_theta=X*theta;
  [m,n]=size(y);
  Jval=sum(y.*log(sigmoid(h_theta))+(1-y).*log(1-sigmoid(h_theta)))/(-m);
  [j,q]=size(theta);
  for i=1:j
    Gradient(i)=X(:,i)'*(sigmoid(X*theta)-y)/m;
  endfor
end
