# Import the Linear Regression module from sklearn 
from sklearn.linear_model import LinearRegression 
import numpy as np 
 
# Need a training dataset, the model will learn how to add numbers from these data 
# We only need 3 training examples: 
#   2+3=5   #   1+5=6   #   6+5=11 
#X = [[2,3],[1,5],[5,6]] 
#Y = [5,6,11] 
X = [[2,3],[1,5],[3,5]] 
Y = [5,6,-2] 
 
# Fit the linear regression model with the training data 
model = LinearRegression() 
model.fit(X,Y) 
Z= np.array([.1,.5])
Z=np.reshape(Z,[1,-1])
# Done! Now we can use predict to sum two numbers 
DT_predict = model.predict(Z) 
print(DT_predict)
