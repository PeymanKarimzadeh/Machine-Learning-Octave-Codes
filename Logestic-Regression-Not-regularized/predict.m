function p=predict(X1,X2,theta)
  X=[1 X1 X2];
  if X*theta>=0
    p='Admitted'
   else
    p='Not Admitted'
  endif
