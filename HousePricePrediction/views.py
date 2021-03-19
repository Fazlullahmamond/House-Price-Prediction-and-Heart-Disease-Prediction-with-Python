
from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def home(request):
    return render(request, 'index.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    data = pd.read_csv("static/USA_Housing.csv")
    data = data.drop(['Address'], axis=1)
    X = data.drop(['Price'], axis=1)
    Y = data['Price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.10)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    var1 = float( request.GET['n1'] )
    var2 = float( request.GET['n2'] )
    var3 = float( request.GET['n3'] )
    var4 = float( request.GET['n4'] )
    var5 = float( request.GET['n5'] )

    pred = model.predict(np.array([var1, var2, var3, var4, var5]).reshape(1, -1))
    pred = round(pred[0])

    price = 'The Predicted Price is $'+str(pred)

    return render(request, 'predict.html', {'result2': price})






def heart(request):
    return render(request, 'heart.html')


def H_result(request):
    data = pd.read_csv("static/heart.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.1, random_state = 1)

    model1 = LogisticRegression(random_state=1) # get instance of model
    model1.fit(x_train, y_train) # Train/Fit model 

    var1 = int( request.GET['n1'] )
    var2 = int( request.GET['n2'] )
    var3 = int( request.GET['n3'] )
    var4 = float( request.GET['n4'] )
    var5 = float( request.GET['n5'] )
    var6 = int( request.GET['n6'] )
    var7 = int( request.GET['n7'] )
    var8 = float( request.GET['n8'] )
    var9 = int( request.GET['n9'] )
    var10 = float( request.GET['n10'] )
    var11 = int( request.GET['n11'] )
    var12 = int( request.GET['n12'] )
    var13 = int( request.GET['n13'] )

    p = model1.predict(np.array([var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11, var12, var13]).reshape(1, -1)) # get y predictions
    if(p[0] == 0):
        p = 'patient not diagnosed with Heart Disease'
    elif(p[0] == 1):
        p = 'patient diagnosed with Heart Disease'
    else:
        p = 'Something went wrong please try again'

    return render(request, 'heart.html', {'result2': p})
