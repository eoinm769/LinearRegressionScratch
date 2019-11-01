import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('Dataset/headbrain.csv')
print(df)

df.describe()
df.info()
df.corr()
df.corr().plot()

X = df.drop(columns ='Brain Weight(grams)')
Y = df['Brain Weight(grams)']

plt.scatter(X['Head Size(cm^3)'],Y)

X = X['Head Size(cm^3)']

from sklearn.model_selection import train_test_split

train_X,test_X,train_Y,test_Y = train_test_split(X,Y,test_size=0.5,train_size=0.5,random_state=48,shuffle=True)

#takes input numpy list of features and target and generates coordiantes
def linearRegressionLine(train_X,train_Y):
    X_mean = float(train_X.mean())
    Y_mean = float(train_Y.mean())

    b1 = 0
    numerator = 0
    denominator = 0

    temp_X = train_X.values.tolist()
    temp_Y = train_Y.values.tolist()

    for ix,iy in zip(temp_X,temp_Y):
        numerator += (ix-X_mean)*(iy-Y_mean)
        denominator += (ix-X_mean)*(ix-X_mean)
    
    b1 = numerator/denominator
    b0 = Y_mean - b1*X_mean

    X_cord = temp_X
    Y_cord = []

    for i in X_cord:
        Y_cord.append(b1*i+b0)

    return X_cord,Y_cord

#takes input the training set feature and target as numpy list and plots the best-fit line and the scatter plot
def plotLineScatter(train_X,train_Y):
	X_cord,Y_cord = linearRegressionLine(train_X,train_Y)
    plt.figure(figsize=(10,10))
    plt.scatter(train_X,train_Y)
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression")
    plt.plot(X_cord,Y_cord,label="linearRegressionLine",color='r',alpha=0.8)

#takes input the training set feature and target as numpy list 
def sumOfSquareOfError(train_X,train_Y):
	X_cord,Y_cord = linearRegressionLine(train_X,train_Y)

	SSE = 0

	for i,j in zip(Y_cord,train_Y):
		SSE += (j-i)**2

	return SSE
#takes input the training set feature and target as numpy list 
def sumOfSquareOfRegression(train_X,train_Y):
	X_cord,Y_cord = linearRegressionLine(train_X,train_Y)
	Y_mean = train_Y.mean()

	SSR = 0

	for i,j in zip(Y_cord,train_Y):
		SSR += (i-Y_mean)**2

	return SSR

#takes input the training set feature and target as numpy list 
def sumOfSquareOfTotal(train_X,train_Y):
	Y_mean = train_Y.mean()

	SST = 0

	for i in train_Y:
		SST += (i-Y_mean)**2

	return SST

#takes input the training set feature and target as numpy list 
def meanSquareError(train_X,train_Y):
	SSE = sumOfSquareOfError(train_X,train_Y)
	n = len(train_X)

	return SSE/(n-2)

#takes input the training set feature and target as numpy list 
def standardErrorOfEstimate(train_X,train_Y):
	return meanSquareError(train_X,train_Y)**0.5

#takes input the training set feature and target as numpy list 
def rSquare(train_X,train_Y):
	return sumOfSquareOfRegression(train_X,train_Y)/sumOfSquareOfTotal(train_X,train_Y)

#takes input the training set feature and target as numpy list 
def standardDeviation(X):
	length = len(X)
	mean_X = X.mean()
	sumSquare = 0

	for i in X:
		sumSquare += (i-mean_X)**2

	deviation = (sumSquare/length-1)**0.5

	return deviation

#takes input the training set feature and target as numpy list 
def correlation(train_X,train_Y,regressionSlope):
	return regressionSlope*(standardDeviation(train_X)/standardDeviation(train_Y))

#takes input the test set feature and regression slope and bias coefficient
def prediction(test_X,regressionSlope,biasCoefficient):
	predict_Y = []

	for i in test_X:
		Y = regressionSlope*i+biasCoefficient
		predict_Y.append(Y)

	return predict_Y

#takes input the prediction value and original value
def MSEOfPredictedRegression(predict_Y,test_Y):
	error = 0

	for i,j in zip(predict_Y,test_Y):
		error += (i-j)**2

	return error/len(predict_Y) 

#takes input the prediction value and original value
def RMSEPredictedRegression(predict_Y,test_Y):
	return MSEOfPredictedRegression(predict_Y,test_Y)**0.5

#takes input the test features and target
def predictionScore(test_X,test_Y):
	SSE = sumOfSquareOfError(test_X,test_Y)
	SST = sumOfSquareOfTotal(test_X,test_Y)

	score = 1-(SSE/SST)

	return score