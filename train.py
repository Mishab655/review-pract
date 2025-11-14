from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
import pandas as pd
import joblib


def train():
    data = load_iris()
    
    
    
    X = data.data
    y = data.target
    
    model = LinearRegression()
    model.fit(X,y)
    joblib.dump(model, 'model.pkl')

if __name__=='__main__':
    train()